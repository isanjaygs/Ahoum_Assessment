# Prompt Log

This document records the material AI interaction history, model selections, design prompts, and corrections made during the development of the Scalable Conversation Facet Evaluator.

---

## Model Selection & Rationale
We chose **`Qwen/Qwen2.5-7B-Instruct`** as the default hosted model (via Hugging Face Serverless API) and **`Qwen/Qwen2.5-1.5B-Instruct`** as the default local fallback model.
* **Why Qwen-2.5?** It represents state-of-the-art instruction-following and structured JSON generation among open-weight models under 16B.
* **Why 1.5B Local Fallback?** Running an 8B model locally on an 8GB Apple M2 Mac causes massive swapping and thermal throttling. The 1.5B parameter version fits comfortably in 3GB of RAM, runs fast on the CPU/MPS (Metal) backend, and provides a fully self-contained offline backup.

---

## Key Prompts & Inputs

### 1. Preprocessing & Taxonomy Audit Prompt
* **Tool/Model:** Gemini 3.5 Flash (Medium)
* **Goal:** Group raw facets from a CSV into clean taxonomy classifications.
* **Prompt Strategy:** Analyze lists of keywords representing physiological measurements, biographical records, daily habits, and cognitive tests to separate observable traits from non-observable metrics.

### 2. Scoring System Prompt (in `src/scoring.py`)
* **Tool/Model:** Gemini 3.5 Flash (Medium)
* **Prompt Content:**
```text
You are an expert psychological and behavioral annotator.
Analyze the following conversation turn or short snippet:
---
[CONVERSATION]
{convo_text}
---

You must evaluate the conversation against these candidate facets:
{facets_bullet_list}

For each candidate facet, decide if there is conversational evidence in the snippet to score it.
Follow these scoring guidelines strictly:
- **1**: Strong evidence of very low / opposite expression (e.g. explicitly stating a strong aversion to risk).
- **2**: Some evidence of low expression.
- **3**: Mixed, moderate, or balanced evidence.
- **4**: Clear evidence of high expression.
- **5**: Strong, repeated, or direct evidence of very high expression.
- If there is no evidence or the evidence is insufficient to score the facet, you MUST set status to "insufficient_evidence", score to null, and explain why.
- Critical Rule: A lack of evidence does NOT map to score 1; it maps to "insufficient_evidence" and score null.
- You must NOT guess or invent scores. Be conservative. If you are unsure or evidence is lacking, abstain.

Respond ONLY with a valid JSON array of objects matching the schema...
```

---

## What AI Got Wrong / What I Corrected

### Example 1: Brittle Substring Matching in Facet Audit
* **Symptom:** Facets like `Originality`, `Courageousness`, `Language use`, and `Well-being component: Engagement` were incorrectly classified as `biographical_demographic` and `not_observable`, and `Types of Mindfulness Techniques Used` was classified as `cognitive_skill` and `not_observable`.
* **Root Cause:** The generated Python code used simple substring containment checks:
  ```python
  if any(k in norm_lower for k in biographical_strong):
  ```
  Since the biographical keywords list contained `"age"` (to match age parameters) and the cognitive list contained `"iq"` (to match IQ), these substring checks matched the `"age"` in `original-ity`, `cour-age-ousness`, `langu-age`, and `eng-age-ment`, and the `"iq"` in `techn-iq-ues`!
* **Correction:** I implemented a robust `contains_keyword` helper using regular expression word boundaries `\b` for short or ambiguous stems:
  ```python
  def contains_keyword(text: str, keyword: str) -> bool:
      bounded_keywords = ['age', 'origin', 'sex', 'pain', 'diet', 'fsh', 'sufi', 'aura', 'input', 'iq']
      if keyword in bounded_keywords:
          return bool(re.search(rf'\b{keyword}\b', text))
      return keyword in text
  ```
* **Verification:** Re-running `src/audit.py` produced 100% accurate classification categories and removed the false positives.

### Example 2: Hugging Face InferenceClient API Signature Error
* **Symptom:** During benchmark execution, the LLM scoring backend crashed with:
  `TypeError: InferenceClient.chat_completion() got an unexpected keyword argument 'timeout'`
* **Root Cause:** The assistant assumed that the `chat_completion` method took a `timeout` keyword argument directly. In `huggingface_hub`, the timeout parameter must be supplied during client initialization (`InferenceClient(..., timeout=timeout)`) rather than at method call time.
* **Correction:** Modified the Hugging Face API client call structure:
  ```python
  client = InferenceClient(
      model=config.LLM_MODEL, 
      token=config.LLM_API_KEY if config.LLM_API_KEY else None,
      timeout=timeout
  )
  response = client.chat_completion(
      messages=messages,
      max_tokens=2048,
      temperature=0.1
  )
  ```
* **Verification:** The benchmark ran successfully, handling API errors and timeouts gracefully without throwing PyTorch or TypeError exceptions.
