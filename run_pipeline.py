import argparse
import sys
import os
from src import config
from src.audit import audit_facets_pipeline
from src.benchmark import run_benchmark

def main():
    parser = argparse.ArgumentParser(
        description="Scalable Conversation Facet Evaluator - Master Orchestrator Pipeline"
    )
    parser.add_argument(
        "--audit-only", 
        action="store_true", 
        help="Run only the facet audit/enrichment preprocessing stage."
    )
    parser.add_argument(
        "--benchmark-only", 
        action="store_true", 
        help="Run only the benchmark evaluation stage."
    )
    
    args = parser.parse_args()
    
    print("==========================================================")
    print("      SCALABLE CONVERSATION FACET EVALUATOR PIPELINE       ")
    print("==========================================================\n")
    
    # Verify environment
    print("--- Environment Config ---")
    print(f"LLM Provider    : {config.LLM_PROVIDER}")
    print(f"LLM Model       : {config.LLM_MODEL}")
    print(f"Timeout / Retry : {config.LLM_TIMEOUT}s / {config.LLM_MAX_RETRIES} retries")
    print(f"Data Dir        : {config.DATA_DIR}\n")
    
    if args.audit_only:
        print(">>> Executing Audit Preprocessing Stage Only...\n")
        audit_facets_pipeline(config.RAW_CSV_PATH, config.ENRICHED_CSV_PATH)
        print("\nAudit Complete.")
        sys.exit(0)
        
    if args.benchmark_only:
        print(">>> Executing Benchmark Stage Only...\n")
        run_benchmark()
        print("\nBenchmark Complete.")
        sys.exit(0)
        
    # Default: Run everything
    print(">>> Executing Preprocessing & Audit Stage...")
    audit_facets_pipeline(config.RAW_CSV_PATH, config.ENRICHED_CSV_PATH)
    print("\n>>> Executing Evaluation Benchmark Stage...\n")
    run_benchmark()
    print("Pipeline Execution Finished.")

if __name__ == '__main__':
    main()
