import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmark_selection import StageResult, select_candidates
from dpc.llm.openai_llm import OpenAILLM
import benchmark_bridge


CANDIDATES = [
    {"candidate_id": f"c{index}", "sql": f"SELECT {index}"} for index in range(4)
]


def stage(sql, *, status="success", api_calls=2, input_tokens=10, output_tokens=4):
    return StageResult(
        status=status,
        selected_sql=sql,
        runtime_ms=12.5,
        usage={
            "api_calls": api_calls,
            "api_successful_calls": api_calls,
            "api_failed_calls": 0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reported_input_tokens": input_tokens,
            "reported_output_tokens": output_tokens,
            "token_usage_complete": True,
        },
    )


class BenchmarkSelectionTest(unittest.TestCase):
    def test_sequential_returns_every_id_once_and_counts_calls(self):
        requested = []
        desired = ["c2", "c0", "c3"]

        def choose(remaining):
            requested.append([candidate["candidate_id"] for candidate in remaining])
            selected_id = desired[len(requested) - 1]
            return stage(next(candidate["sql"] for candidate in remaining if candidate["candidate_id"] == selected_id))

        result = select_candidates(CANDIDATES, choose, "sequential")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["ranking"], ["c2", "c0", "c3", "c1"])
        self.assertEqual(len(set(result["ranking"])), 4)
        self.assertEqual(requested, [["c0", "c1", "c2", "c3"], ["c0", "c1", "c3"], ["c1", "c3"]])
        self.assertNotIn("scores", result)
        self.assertEqual(result["usage"]["selector_invocations"], 3)
        self.assertEqual(result["usage"]["api_calls"], 6)
        self.assertEqual(result["usage"]["input_tokens"], 30)

    def test_one_shot_keeps_native_selector_contract(self):
        result = select_candidates(CANDIDATES, lambda remaining: stage(remaining[2]["sql"]), "one_shot")
        self.assertEqual(result["selected_candidate_id"], "c2")
        self.assertNotIn("ranking", result)
        self.assertEqual(result["usage"]["selector_invocations"], 1)

    def test_duplicate_sql_maps_to_first_remaining_id(self):
        candidates = [dict(candidate) for candidate in CANDIDATES]
        candidates[1]["sql"] = candidates[0]["sql"]
        result = select_candidates(candidates, lambda remaining: stage(remaining[0]["sql"]), "sequential")
        self.assertEqual(result["ranking"], ["c0", "c1", "c2", "c3"])

    def test_partial_failure_preserves_trace_and_zeroes_complete_ranking(self):
        calls = 0

        def choose(remaining):
            nonlocal calls
            calls += 1
            if calls == 2:
                failed = stage(None, status="timeout")
                failed.error = "selector timed out"
                failed.usage["input_tokens"] = None
                failed.usage["token_usage_complete"] = False
                return failed
            return stage(remaining[1]["sql"])

        result = select_candidates(CANDIDATES, choose, "sequential")
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["partial_ranking"], ["c1"])
        self.assertNotIn("ranking", result)
        self.assertEqual(len(result["selection_trace"]), 2)
        self.assertEqual(result["usage"]["selector_invocations"], 2)
        self.assertIsNone(result["usage"]["input_tokens"])
        self.assertFalse(result["usage"]["token_usage_complete"])

    def test_unknown_selection_is_failure_not_first_candidate(self):
        result = select_candidates(CANDIDATES, lambda remaining: stage("SELECT unknown"), "one_shot")
        self.assertEqual(result["status"], "invalid_output")
        self.assertNotIn("selected_candidate_id", result)
        self.assertEqual(result["selection_trace"][0]["selected_candidate_id"], None)

    def test_llm_counts_requests_including_retries(self):
        llm = OpenAILLM(api_key="test", max_retries=2, retry_delay=0)
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3),
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        )
        llm.client.chat.completions.create = Mock(side_effect=[RuntimeError("retry"), response])
        self.assertEqual(llm.ask([{"role": "user", "content": "test"}]), "answer")
        usage = llm.get_usage()
        self.assertEqual(usage["api_calls"], 2)
        self.assertEqual(usage["api_failed_calls"], 1)
        self.assertEqual(usage["api_successful_calls"], 1)
        self.assertEqual(usage["prompt_tokens"], 11)
        prompt_trace = llm.get_prompt_trace()
        self.assertEqual(len(prompt_trace), 2)
        self.assertEqual(prompt_trace[0]["request_status"], "api_error")
        self.assertEqual(prompt_trace[1]["request_status"], "api_response")
        self.assertIsNone(benchmark_bridge._stage_usage(llm)["input_tokens"])
        llm.reset_usage()
        self.assertEqual(llm.get_usage()["api_calls"], 0)
        self.assertEqual(llm.get_prompt_trace(), [])

    def test_bridge_emits_ranking_and_usage_without_labels(self):
        class FakeLLM:
            def __init__(self, **kwargs):
                self.calls = 0

            def reset_usage(self):
                self.calls = 0

            def get_usage(self):
                return {
                    "api_calls": self.calls,
                    "api_successful_calls": self.calls,
                    "api_failed_calls": 0,
                    "usage_reported_calls": self.calls,
                    "prompt_tokens": self.calls * 7,
                    "completion_tokens": self.calls * 2,
                }

            def get_prompt_trace(self):
                return []

        class FakePipeline:
            def __init__(self, **kwargs):
                self.llm = kwargs["llm"]

            def run(self, **kwargs):
                self.llm.calls += 1
                return {"selected_sql": kwargs["candidate_sqls"][-1], "selection_reason": "mock"}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "public.jsonl"
            destination = root / "predictions.jsonl"
            source.write_text(json.dumps({
                "benchmark_id": "toy",
                "question_id": "toy:0",
                "db_id": "toy",
                "question": "Which?",
                "candidates": CANDIDATES,
            }) + "\n", encoding="utf-8")
            args = SimpleNamespace(input=str(source), output=str(destination), db_root=str(root), method_id="dpc_sequential", mode="sequential")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LLM_MODEL": "mock"}), patch.object(benchmark_bridge, "OpenAILLM", FakeLLM), patch.object(benchmark_bridge, "DPCPipeline", FakePipeline), patch.object(benchmark_bridge, "SlicerAgent", lambda llm: None), patch.object(benchmark_bridge, "TesterAgent", lambda llm: None), patch.object(benchmark_bridge, "PythonSolverAgent", lambda llm: None):
                benchmark_bridge.process_benchmark(args)
            output = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(output["ranking"], ["c3", "c2", "c1", "c0"])
            self.assertEqual(output["usage"]["api_calls"], 3)
            self.assertEqual(output["usage"]["selector_invocations"], 3)
            self.assertEqual(output["model_id"], "mock")
            self.assertNotIn("gold_sql", destination.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
