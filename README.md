# Scalable Conversation Facet Evaluator

A compact, production-minded baseline pipeline designed to evaluate conversational text against a large, heterogeneous catalog of facets. The architecture scales conceptually to 5,000+ facets by employing a **dual-pathway routing** mechanism, **hybrid retrieval** (semantic search + lexical expansion), and a **deterministic policy engine** for non-observable facets.

---

## Architecture Summary

This system avoids the expensive and error-prone practice of asking an LLM to evaluate every single facet at once. Instead, it partitions the work into clear deterministic and probabilistic stages:

```
                    RAW FACET CSV
                         │
                         ▼
              ┌──────────────────────┐
              │ AUDIT + NORMALIZATION │
              │   (src/audit.py)     │
              └──────────┬───────────┘
                         ▼
                 ENRICHED CATALOGUE
                         │
           ┌─────────────┼──────────────┐
           │             │              │
           ▼             ▼              ▼
        Direct        Indirect       Non-observable
       observable    observable            │
           │             │                  ▼
           └──────┬──────┘          POLICY ABSTENTION
                  │                   (src/policy.py)
                  ▼                         │
          HYBRID RETRIEVAL                  ▼
        semantic + keywords          status: not_observable
          (src/database.py)          score: null
                  │                         │
                  ▼                         │
             TOP-K FACETS                   │
                  │                         │
                  ▼                         │
            SCORING ENGINE                  │
           (src/scoring.py)                 │
                  │                         │
                  ▼                         │
          SCHEMA VALIDATION                 │
                  │                         │
                  ▼                         ▼
              ┌───────────────────────────────┐
              │       FINAL OUTPUT            │
              │ status, score, conf, evidence │
              └───────────────────────────────┘
```

1. **Preprocessing & Audit (`src/audit.py`):** Cleans raw facet strings and classifies them into a three-tier observability taxonomy:
   * `"direct"`: Style markers immediately present in conversation (e.g., sarcasm, brevity, humor).
   * `"indirect"`: Personality traits/attitudes that can be inferred but require high evidence thresholds (e.g., risk-taking, cooperation).
   * `"not_observable"`: Clinical, biographical, or physiological facts requiring external data (e.g., FSH level, nationality, sleep apnea diagnosis).
2. **Routing & Hybrid Retrieval (`src/database.py`):**
   * `"not_observable"` facets bypass the LLM entirely and go to the Policy Engine.
   * `"direct"` and `"indirect"` facets are indexed in memory. The top-$K$ candidates are retrieved using a blend of semantic vector similarity (`all-MiniLM-L6-v2`) and keyword triggers.
3. **Deterministic Policy Engine (`src/policy.py`):** Instantly resolves unobservable facets to `"status": "not_observable"` and `score: null`, inheriting the audit's `rule_confidence` and pre-defined reason.
4. **LLM Scoring Client & Parser (`src/scoring.py`):** Batches the conversation and the retrieved candidate facets into a single prompt. Parses the JSON output robustly, enforcing strict schema validation and falling back to regex extraction or safe default values in case of failure.

---

## Setup & Installation

The project uses **`uv`** as its Python package installer and manager.

### 1. Create Virtual Environment and Install Dependencies
In the root directory of the workspace, run:
```bash
# Locate uv and create venv
/Users/sanjaygs/.local/bin/uv venv .venv
source .venv/bin/activate

# Install requirements
/Users/sanjaygs/.local/bin/uv pip install -r requirements.txt
```

### 2. Configurations
Define your configurations using environment variables. These are loaded in `src/config.py`:

* `LLM_PROVIDER`: `"huggingface"` (default), `"openai"`, or `"local"`
* `LLM_MODEL`: The Hugging Face model identifier (default: `"Qwen/Qwen2.5-7B-Instruct"`)
* `LLM_API_KEY`: Hugging Face token or OpenAI-compatible provider key (Optional for anonymous HF API runs, but recommended to avoid rate limits).
* `LLM_BASE_URL`: Base URL if using an OpenAI-compatible provider.
* `LLM_TIMEOUT`: Timeout limit in seconds (default: `15.0`).
* `LLM_MAX_RETRIES`: Number of connection retries with exponential backoff (default: `2`).

---

## Run Instructions

You can run components independently or execute the master pipeline.

### Run Preprocessing & Audit
```bash
.venv/bin/python -m src.audit
```
This reads the raw `Facets Assignment.csv` and generates the enriched dataset in `data/facets_enriched.csv`.

### Run Unit Tests
```bash
.venv/bin/python -m unittest discover -s tests
```
This executes all unit tests verifying the preprocessing rules, hybrid retrieval, and output parser.

### Run the Evaluation Benchmark
```bash
.venv/bin/python run_pipeline.py
```
This runs the audit preprocessing, builds the retriever index, evaluates 10 edge case conversations, and outputs a tabular report of the global metrics.

---

## Scaling to 5,000+ Facets

To scale this design to 5,000+ facets, the architecture handles each step as follows:

1. **Indexing & Retrieval:** Precomputing embeddings for 5,000 facets using `all-MiniLM-L6-v2` produces a 5000 × 384 matrix. Cosine similarity in NumPy takes `< 5ms` on CPU. If scaling to 50,000+, we can swap in a lightweight vector index like `hnswlib` or `FAISS` to execute queries in sub-millisecond times.
2. **Deterministic Pre-Routing:** The Policy Engine intercepts and resolves all unobservable facets locally in $\le 1\text{ms}$. If 60% of the 5,000 facets are non-observable, the active candidate pool for LLM scoring drops to 2,000 facets.
3. **LLM Calls & Batching:** Rather than querying the LLM for all facets, we retrieve only the top-$K$ (e.g. $K=15$) observable candidates. This guarantees that only one LLM call is executed per conversation turn, keeping token consumption and latency constant regardless of the total catalog size.
4. **Caching:** Embeddings are cached on disk as a serialized NumPy array (`.npy`), avoiding re-embedding the catalogue at start-up. LLM outputs can be cached using conversation text hashes to bypass API queries for identical dialogues.
5. **Bottleneck Analysis:** The primary bottleneck is the LLM inference latency. Running a local 7B model or calling a hosted API takes ~1-3 seconds. The local preprocessing, vector retrieval, policy routing, and schema parsing execute in `< 15ms` combined.

---

## Known Limitations

1. **Anonymous API Requests:** Running without `HF_TOKEN` causes the Hugging Face serverless client to raise authorization errors. To run LLM scoring successfully, define `HF_TOKEN` (or `LLM_API_KEY`) in your environment.
2. **Local Fallback Accuracy:** The local fallback model `Qwen/Qwen2.5-1.5B-Instruct` fits in 8GB RAM but lacks the reasoning capacity of larger models. It can struggle with sarcasm or complex double negatives compared to 7B or 14B models.

---

## What We Would Improve With Another Day

1. **UI Dashboard:** Build a simple Streamlit interface where users can paste custom text, see retrieved facets, and watch scores, confidence, and text evidence render in real-time.
2. **Confidence Calibration:** Scale predicted confidence levels (`high`/`medium`/`low`) dynamically by comparing the LLM's softmax logprobs, or by performing prompt perturbation (averaging scores across multiple generations).
3. **Structured Outputs (Instructor/Pydantic):** Integrate Pydantic-based output validation (e.g., using Hugging Face's structured JSON schema support) to eliminate raw text JSON parsing and guarantee 100% schema compliance.
