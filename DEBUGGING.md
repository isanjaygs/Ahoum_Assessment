# Debugging Log

This document records the real bugs and failed assumptions encountered during development and testing, tracking their symptom, diagnosis, root cause, fix, and verification.

---

## Issue 1: Brittle Substring Matching in Facet Audit

### Symptom
Facets like `Originality`, `Courageousness`, `Language use`, and `Well-being component: Engagement` were incorrectly classified as `biographical_demographic` / `not_observable`, and `Types of Mindfulness Techniques Used` was classified as `cognitive_skill` / `not_observable`.

### Diagnosis & Root Cause
The taxonomy classifier used a simple substring inclusion check:
```python
if any(k in norm_lower for k in biographical_strong):
```
Because the biographical list contained `"age"` (to capture age parameters) and the cognitive list contained `"iq"` (to capture IQ tests), these short strings matched inside larger words:
* `"age"` matched `original-ity`, `cour-age-ousness`, `langu-age`, and `eng-age-ment`.
* `"iq"` matched `techn-iq-ues`.

### Fix
Created a helper function `contains_keyword` that enforces regular expression word boundaries `\b` for short or ambiguous keywords, preventing partial substring matches:
```python
def contains_keyword(text: str, keyword: str) -> bool:
    bounded_keywords = [
        'age', 'origin', 'sex', 'pain', 'diet', 'fsh', 'sufi', 'aura', 'input', 'iq'
    ]
    if keyword in bounded_keywords:
        return bool(re.search(rf'\b{keyword}\b', text))
    return keyword in text
```

### Verification
Ran `src/audit.py` and printed category counts. The false positives vanished:
* `Originality` and `Courageousness` default to `personality_trait` / `indirect` (observable).
* `Language use` defaults to `personality_trait` / `direct` (observable).
* `Mindfulness Techniques Used` correctly classified under `spiritual_religious` / `not_observable` (instead of cognitive).

---

## Issue 2: Hugging Face InferenceClient API Signature Error

### Symptom
During benchmark execution, the LLM scoring client crashed with:
`TypeError: InferenceClient.chat_completion() got an unexpected keyword argument 'timeout'`

### Diagnosis & Root Cause
The assistant assumed that `InferenceClient.chat_completion()` accepts a `timeout` argument. In the current `huggingface_hub` Python package version, the `timeout` parameter must be passed during client instantiation (`InferenceClient(..., timeout=timeout)`) rather than at method call time.

### Fix
Removed the `timeout` argument from `chat_completion(...)` and passed it to `InferenceClient()` instantiation instead:
```python
client = InferenceClient(
    model=config.LLM_MODEL, 
    token=config.LLM_API_KEY if config.LLM_API_KEY else None,
    timeout=timeout
)
```

### Verification
Re-ran `run_pipeline.py`. The pipeline completed successfully and handled timeouts/connection errors cleanly without throwing TypeErrors.

---

## Issue 3: Missing Facet in Enriched Catalogue during Retrieval Benchmark

### Symptom
Retrieval recall on Case 7 was 0.0% and Case 3 was 66.7% because the expected facet `Feeling energetic` was missing from candidate pools.

### Diagnosis & Root Cause
We specified `Feeling energetic` as one of our benchmark's 20 representative facets. However, there was no facet named `Feeling energetic` in the original `Facets Assignment.csv`. Since it was not found, the database routed it as a non-observable fallback facet (and bypassed retrieval indexing), causing it to never show up in retrieved candidates.

### Fix
Found that the CSV actually contains the facet **`High-spiritedness`**. I replaced `"Feeling energetic"` with `"High-spiritedness"` in both `src/benchmark.py` and the database lexical rules.

### Verification
Re-ran `run_pipeline.py`. The global retrieval recall reached **100% (17/17)** on the observable benchmark targets.
