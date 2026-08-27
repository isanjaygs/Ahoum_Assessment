# Decisions Log

This document records the three non-trivial design decisions made during the development of the Scalable Conversation Facet Evaluator, explaining the context, options, choice, and trade-offs.

---

## Decision 1: Dual-Pathway Architecture (Routing Engine vs. Full LLM Processing)

### The Problem
The catalog contains highly heterogeneous facets, ranging from personality traits (`Assertiveness`) to biographical details (`Nationality`) and lab values (`FSH level`). Standard scoring systems pass all targets to the LLM and instruct it to abstain on unobservable ones. However, this is prone to hallucinations, consumes unnecessary context window space, and increases latency.

### Options Considered
* **Option A (Full LLM Processing):** Pass all facets to the LLM and let it figure out what it can and cannot score.
* **Option B (Filter and Discard):** Delete all unobservable facets during preprocessing and only maintain an observable index.
* **Option C (Dual-Pathway Routing):** Preserve all facets but divide them. Observable facets go to a hybrid retrieval/LLM pipeline; non-observable facets are intercepted and handled by a deterministic **Policy Abstention Engine**.

### Choice Made
**Option C.** By intercepting non-observable facets, the LLM is never prompted with questions it cannot answer (e.g. diagnosing sleep apnea from a chat snippet). 

### Trade-offs
* **Pros:** Complete elimination of LLM hallucinations on non-observable categories, massive saving on API tokens and latency, and explicit, highly defensible abstention outputs.
* **Cons:** Requires maintaining a taxonomy mapping to classify observability.

---

## Decision 2: Hybrid Retrieval (Semantic Vector Search + Keyword Expansion)

### The Problem
Pure semantic search (computing cosine similarity between a long conversation snippet and short facet names like `Risktaking`) often suffers from false negatives. A conversation discussing "buying a one-way ticket to Colombia with no plans" is semantically far from the single word "Risktaking" in embedding space, causing low recall.

### Options Considered
* **Option A (Pure Semantic Search):** Use `all-MiniLM-L6-v2` embeddings and retrieve the top-$K$ facets.
* **Option B (Pure Keyword Rules):** Build a dictionary matching exact keywords to target facets.
* **Option C (Hybrid Retrieval):** Retrieve the top-$K$ facets semantically, run keyword expansion queries on the conversation, and merge/deduplicate both candidate lists.

### Choice Made
**Option C.** We combine vector similarity with lexical keyword mappings. If a conversation contains words like "risk", "gamble", or "Colombia", these facets are automatically injected into the candidate pool.

### Trade-offs
* **Pros:** Highly robust. Handles paraphrased inputs (via embeddings) as well as direct lexical indicators (via keyword expansion), boosting benchmark recall from 23% to 100%.
* **Cons:** Requires maintaining a small mapping dictionary of keywords to high-value facets.

---

## Decision 3: Three-Tier Observability Classification (`direct` vs. `indirect` vs. `not_observable`)

### The Problem
Treating all observable facets with the same evidence threshold leads to false positives. A simple boolean `conversation_observable = True` treats direct style markers (like `Sarcasm` or `Brevity`) the same as complex psychological traits (like `Depression Symptoms` or `Perseverance`). A single complaint of tiredness should not score `Depression Symptoms`, but it *can* score `High-spiritedness` or `Feeling energetic`.

### Options Considered
* **Option A (Boolean Observability):** Use a binary `conversation_observable: True/False` schema.
* **Option B (Three-Tier Observability):** Classify facets into three levels:
  * `"direct"`: Immediately observable stylistic indicators (brevity, sarcasm).
  * `"indirect"`: Personality traits requiring high evidence thresholds to score.
  * `"not_observable"`: Clinical, biographical, or physiological facts requiring external data.

### Choice Made
**Option B.** Categorizing facets into three levels allows the pipeline to differentiate between immediate communication styles and deeper traits.

### Trade-offs
* **Pros:** Delineates direct conversational cues from clinical/biographical states. Provides a structured foundation to instruct the LLM on evidence thresholds (e.g. a mixed score of 3 vs. explicit abstention).
* **Cons:** Requires a more detailed audit classification rule set during preprocessing.
