import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

# Add DPC root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DPC-Bridge")

from dpc.llm.openai_llm import OpenAILLM
from dpc.agents.slicer_agent import SlicerAgent
from dpc.agents.tester_agent import TesterAgent
from dpc.agents.solver_agent import PythonSolverAgent
from dpc.core.pipeline import DPCPipeline
from dpc.utils.clustering import cluster_sql_candidates

def process_benchmark(args):
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("LLM_MODEL", "gpt-4o")
    
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment. DPC might fail if it uses the LLM API.")
        
    llm = OpenAILLM(model_name=model_name, api_key=api_key)
    pipeline = DPCPipeline(
        slicer=SlicerAgent(llm),
        tester=TesterAgent(llm),
        solver=PythonSolverAgent(llm),
        grouper=None, # We use execution mode
        llm=llm
    )
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip():
                continue
                
            data = json.loads(line)
            question_id = data["question_id"]
            db_id = data["db_id"]
            question = data["question"]
            candidates = data["candidates"] # list of {"candidate_id": "...", "sql": "..."}
            
            db_path = str(Path(args.db_root) / db_id / f"{db_id}.sqlite")
            
            sqls = [c["sql"] for c in candidates]
            
            start_time = time.time()
            usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "api_calls": 0}
            
            status = "success"
            error_msg = ""
            selected_sql = None
            
            try:
                # 1. Run DPC Pipeline
                # Using defaults: phase1_selection_mode="execution", eval_metric="bs_f1", num_test_data=3, num_solver_attempts=3
                result = pipeline.run(
                    question=question,
                    db_path=db_path,
                    candidate_sqls=sqls,
                    sql_timeout=30,
                    python_timeout=30,
                    num_test_data=3,
                    num_solver_attempts=3,
                    phase1_selection_mode="execution",
                    eval_metric="bs_f1"
                )
                selected_sql = result["selected_sql"]
                
                # Aggregate token usage
                usage = result.get("token_usage", {})
                usage_totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
                usage_totals["completion_tokens"] += usage.get("completion_tokens", 0)
                usage_totals["api_calls"] += 1 # We just count 1 pipeline run as 1 logic unit, but actual LLM calls might be more. We'll just pass tokens.
                
            except Exception as e:
                logger.error(f"Error processing {question_id}: {e}")
                status = "runtime_error"
                error_msg = str(e)
                
            end_time = time.time()
            runtime_ms = (end_time - start_time) * 1000
            
            output = {
                "schema_version": 1,
                "benchmark_id": data["benchmark_id"],
                "question_id": question_id,
                "method_id": args.method_id,
                "method_type": "selector",
                "status": status,
                "runtime_ms": runtime_ms,
            }
            
            if status == "success":
                # Map selected_sql back to candidate_id
                selected_cid = None
                for c in candidates:
                    if c["sql"] == selected_sql:
                        selected_cid = c["candidate_id"]
                        break
                
                # Fallback if strict string matching fails (e.g., whitespace formatting changes)
                if not selected_cid:
                    selected_cid = candidates[0]["candidate_id"]
                    
                output["selected_candidate_id"] = selected_cid
                output["usage"] = {
                    "api_calls": usage_totals["api_calls"],
                    "input_tokens": usage_totals["prompt_tokens"],
                    "output_tokens": usage_totals["completion_tokens"],
                    "estimated_cost_usd": 0.0 # DPC doesn't calculate cost internally
                }
            else:
                output["error"] = {"message": error_msg}
                
            fout.write(json.dumps(output) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--db-root", required=True)
    args = parser.parse_args()
    
    process_benchmark(args)
