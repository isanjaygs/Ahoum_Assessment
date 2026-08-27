import json
import os
import pandas as pd
from src.database import FacetDatabase
from src.policy import score_non_observable_facets
from src.scoring import score_observable_facets
from src import config

# The 20 representative facets selected for evaluation (10 observable, 10 non-observable)
BENCHMARK_FACETS = [
    # Observable (direct/indirect)
    "Risktaking",
    "Adventure-Seeking Behavior",
    "Creative risk-taking tendency",
    "Merriness",
    "Drollness",
    "High-spiritedness",
    "Perseverance",
    "Language use",
    "Unassertiveness",
    "Peacefulness",
    
    # Non-Observable (routed to policy)
    "FSH level",
    "Clinical depression diagnosis",
    "Sleep Apnea",
    "Nationality",
    "Passport-stamps count",
    "Yoga discipline hours / week",
    "Types of Mindfulness Techniques Used",
    "Quran khatam cycles per year",
    "Public-transport km/week",
    "Intelligence Quotient (IQ)"
]

# The 10 test conversations capturing edge cases and hallucination tests
BENCHMARK_DATA = [
    {
        "id": 1,
        "description": "Clear risk-taking & adventure case",
        "conversation": "I quit my stable corporate job last week and bought a one-way ticket to Colombia with no plans or place to stay. Let's see what happens!",
        "expected_observable_retrieved": ["Risktaking", "Adventure-Seeking Behavior", "Creative risk-taking tendency"],
        "expected_scores": {
            "Risktaking": {"status": "scored", "score": 5},
            "Adventure-Seeking Behavior": {"status": "scored", "score": 5},
            "Creative risk-taking tendency": {"status": "scored", "score": 5}
        }
    },
    {
        "id": 2,
        "description": "Sarcasm / irony analysis",
        "conversation": "Oh, fantastic. I absolutely love sitting in gridlock traffic for two hours every single day, it is literally my dream job commute.",
        "expected_observable_retrieved": ["Drollness", "Merriness"],
        "expected_scores": {
            "Drollness": {"status": "scored", "score": 4},
            "Merriness": {"status": "scored", "score": 1} # Sarcastic happy is low merriness
        }
    },
    {
        "id": 3,
        "description": "Code-switching & perseverance",
        "conversation": "Honestly, me siento muy cansado today. I didn't sleep well at all, but we still have to finish this project. Vamos a darle.",
        "expected_observable_retrieved": ["High-spiritedness", "Perseverance", "Language use"],
        "expected_scores": {
            "High-spiritedness": {"status": "scored", "score": 1},
            "Perseverance": {"status": "scored", "score": 4},
            "Language use": {"status": "scored", "score": 5}
        }
    },
    {
        "id": 4,
        "description": "Quoted/Third-party speech (avoiding wrong inference)",
        "conversation": "My boss literally stood up in the meeting and yelled, 'You are all completely incompetent and lazy!' It was crazy.",
        "expected_observable_retrieved": ["Perseverance", "Unassertiveness"],
        "expected_scores": {
            "Perseverance": {"status": "insufficient_evidence", "score": None},
            "Unassertiveness": {"status": "insufficient_evidence", "score": None}
        }
    },
    {
        "id": 5,
        "description": "Ambiguity / double negatives",
        "conversation": "I wouldn't say I'm not open to new ideas, but I'm certainly not going to just jump into anything without checking the facts first.",
        "expected_observable_retrieved": ["Risktaking", "Creative risk-taking tendency"],
        "expected_scores": {
            "Risktaking": {"status": "scored", "score": 2}, # Cautious
            "Creative risk-taking tendency": {"status": "scored", "score": 2}
        }
    },
    {
        "id": 6,
        "description": "Sarcastic compliance / unassertiveness",
        "conversation": "Sure, let's keep talking over me. I'm sure my input is completely worthless anyway.",
        "expected_observable_retrieved": ["Unassertiveness", "Drollness"],
        "expected_scores": {
            "Unassertiveness": {"status": "scored", "score": 4},
            "Drollness": {"status": "scored", "score": 4}
        }
    },
    {
        "id": 7,
        "description": "Hallucination Test - Tiredness (Medical)",
        "conversation": "I've been feeling extremely fatigued and sluggish for the past few weeks, waking up multiple times during the night.",
        "expected_observable_retrieved": ["High-spiritedness"],
        "expected_scores": {
            "High-spiritedness": {"status": "scored", "score": 1},
            "Sleep Apnea": {"status": "not_observable", "score": None},
            "FSH level": {"status": "not_observable", "score": None},
            "Clinical depression diagnosis": {"status": "not_observable", "score": None}
        }
    },
    {
        "id": 8,
        "description": "Hallucination Test - Pasta (Biographical)",
        "conversation": "I absolutely love cooking. I make a huge batch of fresh pasta from scratch almost every single Sunday evening.",
        "expected_observable_retrieved": [],
        "expected_scores": {
            "Nationality": {"status": "not_observable", "score": None},
            "Passport-stamps count": {"status": "not_observable", "score": None}
        }
    },
    {
        "id": 9,
        "description": "Hallucination Test - Mindfulness (Habits)",
        "conversation": "Lately I've been trying to live a much more mindful, peaceful life, and really focus on being present in each moment.",
        "expected_observable_retrieved": ["Peacefulness"],
        "expected_scores": {
            "Peacefulness": {"status": "scored", "score": 4},
            "Yoga discipline hours / week": {"status": "not_observable", "score": None},
            "Types of Mindfulness Techniques Used": {"status": "not_observable", "score": None}
        }
    },
    {
        "id": 10,
        "description": "Low evidence general chitchat",
        "conversation": "So, did you see the weather forecast for tomorrow? They said it might rain in the afternoon.",
        "expected_observable_retrieved": ["Brevity"],
        "expected_scores": {
            "Brevity": {"status": "scored", "score": 4},
            "Risktaking": {"status": "insufficient_evidence", "score": None},
            "Quran khatam cycles per year": {"status": "not_observable", "score": None}
        }
    }
]

def run_scoring_pipeline(convo_text: str, target_facets: list[str], db: FacetDatabase) -> list[dict]:
    """
    Runs the pipeline in Explicit Facet Evaluation Mode:
    1. Routes explicit facets into observable and non-observable.
    2. Runs Policy Engine on non-observable.
    3. Runs LLM Scoring Engine on observable.
    4. Merges and returns results.
    """
    observable_facets, non_observable_facets = db.route_facets(target_facets)
    
    # Run deterministic policy abstention
    policy_results = score_non_observable_facets(non_observable_facets)
    
    # Run LLM scoring on observable facets
    llm_results = score_observable_facets(convo_text, observable_facets)
    
    return policy_results + llm_results

def run_benchmark():
    print("\n  ┌─────────────────────────────────────────────┐")
    print(  "  │        SCORING BASELINE BENCHMARK           │")
    print(  "  └─────────────────────────────────────────────┘\n")
    
    db = FacetDatabase()
    
    results = []
    
    total_expected_retrieved = 0
    total_actual_retrieved = 0
    
    total_policy_evaluated = 0
    total_policy_correct = 0
    
    total_scoring_evaluated = 0
    total_scoring_correct = 0
    
    for case in BENCHMARK_DATA:
        convo_id = case["id"]
        convo_text = case["conversation"]
        desc = case["description"]
        expected_retrieved = case["expected_observable_retrieved"]
        expected_scores = case["expected_scores"]
        
        snippet = convo_text[:72] + "..." if len(convo_text) > 72 else convo_text
        print(f"  [{convo_id:02d}] {desc}")
        print(f"       {snippet}")
        
        # 1. Evaluate Retrieval (Recall@K)
        # In normal mode, we would retrieve top-K. Let's run retrieval to check if the 
        # expected observable facets are caught in the top-10 candidate pool.
        retrieved_candidates = db.retrieve_observable_facets(convo_text, k=10)
        retrieved_names = {c['normalized_facet'].lower() for c in retrieved_candidates}
        
        case_retrieved = 0
        case_expected_retrieved = len(expected_retrieved)
        
        for f in expected_retrieved:
            total_expected_retrieved += 1
            if f.lower() in retrieved_names:
                case_retrieved += 1
                total_actual_retrieved += 1
                
        recall_pct = (case_retrieved / case_expected_retrieved * 100) if case_expected_retrieved > 0 else 100.0
        
        # 2. Run Pipeline (Explicit Mode)
        # Evaluate scoring over the 20 benchmark facets
        pipeline_outputs = run_scoring_pipeline(convo_text, BENCHMARK_FACETS, db)
        output_map = {o['facet'].lower(): o for o in pipeline_outputs}
        
        case_policy_correct = 0
        case_policy_total = 0
        case_score_correct = 0
        case_score_total = 0
        
        case_results = []
        
        for f_name, exp in expected_scores.items():
            f_key = f_name.lower()
            pred = output_map.get(f_key, None)
            
            is_correct = False
            pred_status = pred['status'] if pred else 'missing'
            pred_score = pred['score'] if pred else None
            
            if exp['status'] == 'not_observable':
                # Policy routing check
                case_policy_total += 1
                total_policy_evaluated += 1
                if pred_status == 'not_observable' and pred_score is None:
                    case_policy_correct += 1
                    total_policy_correct += 1
                    is_correct = True
            else:
                # Scoring check (LLM)
                case_score_total += 1
                total_scoring_evaluated += 1
                if pred_status == exp['status'] and pred_score == exp['score']:
                    case_score_correct += 1
                    total_scoring_correct += 1
                    is_correct = True
                    
            case_results.append({
                "facet": f_name,
                "expected": exp,
                "predicted": {
                    "status": pred_status,
                    "score": pred_score,
                    "confidence": pred['confidence'] if pred else 'low',
                    "evidence": pred['evidence'] if pred else ''
                },
                "is_correct": is_correct
            })
            
        r_icon = "✓" if case_retrieved == case_expected_retrieved else "✗"
        p_str  = f"{case_policy_correct}/{case_policy_total} policy" if case_policy_total > 0 else ""
        s_str  = f"{case_score_correct}/{case_score_total} scored"   if case_score_total  > 0 else ""
        parts  = [x for x in [p_str, s_str] if x]
        print(f"       {r_icon} Retrieval {case_retrieved}/{case_expected_retrieved}  │  " + "  │  ".join(parts))
        print()
        
        results.append({
            "convo_id": convo_id,
            "description": desc,
            "conversation": convo_text,
            "retrieval_recall": recall_pct,
            "evaluations": case_results
        })
        
    # Calculate global metrics
    global_recall = (total_actual_retrieved / total_expected_retrieved * 100) if total_expected_retrieved > 0 else 100.0
    global_policy_acc = (total_policy_correct / total_policy_evaluated * 100) if total_policy_evaluated > 0 else 100.0
    global_scoring_acc = (total_scoring_correct / total_scoring_evaluated * 100) if total_scoring_evaluated > 0 else 100.0
    
    # Save results to data folder
    os.makedirs(os.path.dirname(config.ENRICHED_CSV_PATH), exist_ok=True)
    report_path = os.path.join(config.DATA_DIR, "benchmark_results.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
        
    def _bar(correct, total):
        if total == 0: return "n/a"
        pct = correct / total * 100
        filled = int(pct / 10)
        bar = "█" * filled + "░" * (10 - filled)
        return f"{bar} {correct}/{total} ({pct:.0f}%)"

    print("  ┌──────────────────────────────────────────────────────────┐")
    print(  "  │                   BENCHMARK SUMMARY                     │")
    print(  "  ├──────────────────────────────────────────────────────────┤")
    print(f"  │  Conversations        {len(BENCHMARK_DATA):<36}│")
    print(f"  │  Retrieval Recall@10  {_bar(total_actual_retrieved, total_expected_retrieved):<36}│")
    print(f"  │  Policy Abstentions   {_bar(total_policy_correct, total_policy_evaluated):<36}│")
    print(f"  │  LLM Scoring          {_bar(total_scoring_correct, total_scoring_evaluated):<36}│")
    print(  "  └──────────────────────────────────────────────────────────┘")
    print(f"  Results → {report_path}\n")

if __name__ == '__main__':
    run_benchmark()
