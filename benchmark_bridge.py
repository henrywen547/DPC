"""Run DPC's native selector once or induce a total order by repeated selection.

Only public benchmark rows enter this process. Group sizes and duel scores from
DPC are diagnostic evidence for each selection, never candidate ranking scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from benchmark_selection import StageResult, select_candidates
from dpc.agents.slicer_agent import SlicerAgent
from dpc.agents.solver_agent import PythonSolverAgent
from dpc.agents.tester_agent import TesterAgent
from dpc.core.pipeline import DPCPipeline
from dpc.llm.openai_llm import OpenAILLM


logger = logging.getLogger("DPC-Bridge")


def _stage_usage(llm: OpenAILLM) -> dict:
    observed = llm.get_usage()
    complete = (
        observed["api_failed_calls"] == 0
        and observed["usage_reported_calls"] == observed["api_successful_calls"]
    )
    return {
        "api_calls": observed["api_calls"],
        "api_successful_calls": observed["api_successful_calls"],
        "api_failed_calls": observed["api_failed_calls"],
        "input_tokens": observed["prompt_tokens"] if complete else None,
        "output_tokens": observed["completion_tokens"] if complete else None,
        "reported_input_tokens": observed["prompt_tokens"],
        "reported_output_tokens": observed["completion_tokens"],
        "token_usage_complete": complete,
    }


def _choose(pipeline: DPCPipeline, llm: OpenAILLM, question: str, db_path: str, candidates) -> StageResult:
    llm.reset_usage()
    started = time.monotonic()
    try:
        result = pipeline.run(
            question=question,
            db_path=db_path,
            candidate_sqls=[candidate["sql"] for candidate in candidates],
            sql_timeout=30,
            python_timeout=30,
            num_test_data=3,
            num_solver_attempts=3,
            phase1_selection_mode="execution",
            eval_metric="bs_f1",
        )
        if not isinstance(result, dict):
            return StageResult(
                status="invalid_output",
                runtime_ms=(time.monotonic() - started) * 1000,
                usage=_stage_usage(llm),
                error="DPC returned a non-object result",
                evidence={"prompt_trace": llm.get_prompt_trace()},
            )
        reason = result.get("selection_reason")
        evidence = {
            "selection_reason": reason,
            "champion_score": result.get("champion_score"),
            "challenger_score": result.get("challenger_score"),
            "pipeline_fallback": isinstance(reason, str) and (
                reason.startswith("Slicer Error") or reason.startswith("Verification Error")
            ),
            "prompt_trace": llm.get_prompt_trace(),
        }
        return StageResult(
            status="success",
            runtime_ms=(time.monotonic() - started) * 1000,
            usage=_stage_usage(llm),
            selected_sql=result.get("selected_sql"),
            evidence=evidence,
        )
    except Exception as error:
        status = "timeout" if isinstance(error, TimeoutError) else "runtime_error"
        logger.exception("DPC selection failed")
        return StageResult(
            status=status,
            runtime_ms=(time.monotonic() - started) * 1000,
            usage=_stage_usage(llm),
            error=str(error),
            evidence={"prompt_trace": llm.get_prompt_trace()},
        )


def process_benchmark(args) -> None:
    load_dotenv()
    model_name = os.getenv("LLM_MODEL", "gpt-4o")
    custom_base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        llm = OpenAILLM(model_name=model_name, api_key=api_key)
        pipeline = DPCPipeline(
            slicer=SlicerAgent(llm),
            tester=TesterAgent(llm),
            solver=PythonSolverAgent(llm),
            grouper=None,
            llm=llm,
        )
    else:
        logger.error("OPENAI_API_KEY is missing; every example will be marked runtime_error")
        llm = None
        pipeline = None

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, "r", encoding="utf-8") as source, open(
        output_path, "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            public = json.loads(line)
            db_id = public["db_id"]
            db_path = str(Path(args.db_root) / db_id / f"{db_id}.sqlite")
            started = time.monotonic()

            def choose(remaining):
                if pipeline is None or llm is None:
                    return StageResult(
                        status="runtime_error",
                        runtime_ms=0.0,
                        usage={
                            "api_calls": 0,
                            "api_successful_calls": 0,
                            "api_failed_calls": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "reported_input_tokens": 0,
                            "reported_output_tokens": 0,
                            "token_usage_complete": True,
                        },
                        error="OPENAI_API_KEY is missing",
                        selector_invoked=False,
                    )
                return _choose(pipeline, llm, public["question"], db_path, remaining)

            selection = select_candidates(public["candidates"], choose, args.mode)
            output = {
                "schema_version": 1,
                "benchmark_id": public["benchmark_id"],
                "question_id": public["question_id"],
                "method_id": args.method_id,
                "method_type": "selector" if args.mode == "one_shot" else "ranker",
                "method_mode": "native_selector" if args.mode == "one_shot" else "sequential_ranking",
                "status": selection.pop("status"),
                "model_id": model_name,
                "api_backend": "openai_compatible" if custom_base_url else "openai_default",
                "api_base_url_sha256": hashlib.sha256(custom_base_url.encode("utf-8")).hexdigest()
                if custom_base_url else None,
                "runtime_ms": (time.monotonic() - started) * 1000,
                **selection,
            }
            destination.write(json.dumps(output, ensure_ascii=False) + "\n")
            destination.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--db-root", required=True)
    parser.add_argument("--mode", choices=("one_shot", "sequential"), default="one_shot")
    process_benchmark(parser.parse_args())
