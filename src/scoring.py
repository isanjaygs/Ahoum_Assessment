import json
import re
import time
from src import config

# We use httpx directly for standard OpenAI calls (httpx is installed in the venv)
import httpx

# Lazily import transformers only if local provider is chosen to avoid overhead
local_pipeline = None

def _get_local_pipeline():
    global local_pipeline
    if local_pipeline is not None:
        return local_pipeline
        
    print(f"Loading local model {config.LOCAL_FALLBACK_MODEL}...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using local torch device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_FALLBACK_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.LOCAL_FALLBACK_MODEL,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        device_map="auto"
    )
    local_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        max_new_tokens=1500,
        temperature=0.1,
        do_sample=False
    )
    return local_pipeline

def call_hf_api(messages: list, timeout: float = config.LLM_TIMEOUT) -> str:
    """Calls Hugging Face Serverless Inference API."""
    from huggingface_hub import InferenceClient
    
    client = InferenceClient(
        model=config.LLM_MODEL, 
        token=config.LLM_API_KEY if config.LLM_API_KEY else None,
        timeout=timeout
    )
    
    # We will format using the client's chat completion interface
    response = client.chat_completion(
        messages=messages,
        max_tokens=2048,
        temperature=0.1
    )
    return response.choices[0].message.content

def call_openai_api(messages: list, timeout: float = config.LLM_TIMEOUT) -> str:
    """Calls standard OpenAI-compatible API using httpx."""
    if not config.LLM_BASE_URL:
        raise ValueError("LLM_BASE_URL must be specified for openai provider")
        
    url = f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    if config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"
        
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 2048
    }
    
    response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

def call_local_model(messages: list) -> str:
    """Generates completions using a local transformers pipeline."""
    pipe = _get_local_pipeline()
    # Format messages using the template if supported, else join them
    try:
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback manual format
        prompt = ""
        for m in messages:
            prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
    out = pipe(prompt)
    generated_text = out[0]['generated_text']
    # Extract assistant response only
    if prompt in generated_text:
        return generated_text.replace(prompt, "").strip()
    return generated_text.strip()

def request_llm_scoring(messages: list, retries: int = config.LLM_MAX_RETRIES, timeout: float = config.LLM_TIMEOUT) -> str:
    """Executes the API call with retries and exponential backoff."""
    delay = 2.0
    last_err = None
    
    for attempt in range(retries + 1):
        try:
            if config.LLM_PROVIDER == "huggingface":
                return call_hf_api(messages, timeout)
            elif config.LLM_PROVIDER == "openai":
                return call_openai_api(messages, timeout)
            elif config.LLM_PROVIDER == "local":
                return call_local_model(messages)
            else:
                raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")
        except Exception as e:
            last_err = e
            if attempt < retries:
                print(f"[API Attempt {attempt+1} Failed]: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"[API Call Failed after {retries+1} attempts]: {e}")
                
    raise last_err

def generate_prompt(convo_text: str, candidates: list[dict]) -> list:
    """Generates the chat message structure for LLM scoring."""
    facets_bullet_list = ""
    for f in candidates:
        facets_bullet_list += f"- **{f['normalized_facet']}** ({f['facet_type']})\n"
        
    system_instruction = f"""You are an expert psychological and behavioral annotator.
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

You must respond ONLY with a valid JSON array of objects. Do not wrap it in markdown block tags, and do not include extra explanations outside the JSON array.
Each object in the array must match this exact schema:
{{
    "facet": "<exact normalized facet name>",
    "status": "scored" | "insufficient_evidence",
    "score": 1 | 2 | 3 | 4 | 5 | null,
    "confidence": "high" | "medium" | "low",
    "evidence": "<direct quote or short explanation of reasoning>"
}}
"""
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": "Please analyze the conversation and output the JSON evaluation array."}
    ]

def clean_model_output(output_text: str) -> str:
    """Strips markdown fences or extra leading/trailing whitespace."""
    text = output_text.strip()
    # Remove markdown code blocks if present
    code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        text = code_block_match.group(1).strip()
    # Strip any leading [ or trailing ] if there's text outside it
    first_bracket = text.find('[')
    last_bracket = text.rfind(']')
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        text = text[first_bracket:last_bracket+1]
    return text

def parse_with_regex(raw_text: str) -> list[dict]:
    """Fallback parser that uses regex to find and parse individual JSON objects."""
    results = []
    # Find anything that looks like { ... }
    matches = re.findall(r'\{[^{}]*\}', raw_text)
    for m in matches:
        try:
            obj = json.loads(m)
            if 'facet' in obj:
                results.append(obj)
        except Exception:
            pass
    return results

def validate_and_standardize_results(parsed_list: list, candidates: list[dict]) -> list[dict]:
    """
    Enforces strict schema validation on the parsed objects.
    Ensures:
    1. All target candidate facets are represented in the output.
    2. If status is 'scored', score is an integer 1-5.
    3. If status is 'insufficient_evidence', score is null.
    4. Invalid entries are mapped to 'invalid_model_output'.
    """
    candidate_map = {c['normalized_facet'].lower(): c for c in candidates}
    result_map = {}
    
    for item in parsed_list:
        if not isinstance(item, dict) or 'facet' not in item:
            continue
            
        f_name = str(item['facet']).strip()
        f_key = f_name.lower()
        
        if f_key not in candidate_map:
            continue
            
        status = str(item.get('status', 'insufficient_evidence')).strip().lower()
        score = item.get('score', None)
        confidence = str(item.get('confidence', 'medium')).strip().lower()
        evidence = str(item.get('evidence', '')).strip()
        
        # Validation rules
        if status == 'scored':
            try:
                # Cast to float then int to handle floats like 4.0
                score_val = int(float(score))
                if score_val in [1, 2, 3, 4, 5]:
                    score = score_val
                else:
                    status = 'insufficient_evidence'
                    score = None
                    evidence += " [Schema Alert: Invalid score value forced abstention]"
            except Exception:
                status = 'insufficient_evidence'
                score = None
                evidence += " [Schema Alert: Non-numeric score forced abstention]"
        else:
            status = 'insufficient_evidence'
            score = None
            
        if confidence not in ['high', 'medium', 'low']:
            confidence = 'medium'
            
        result_map[f_key] = {
            'facet': candidate_map[f_key]['normalized_facet'],
            'status': status,
            'score': score,
            'confidence': confidence,
            'evidence': evidence
        }
        
    # Fill in any missing candidate facets with fallback invalid_model_output / abstention
    final_results = []
    for c in candidates:
        c_key = c['normalized_facet'].lower()
        if c_key in result_map:
            final_results.append(result_map[c_key])
        else:
            final_results.append({
                'facet': c['normalized_facet'],
                'status': 'invalid_model_output',
                'score': None,
                'confidence': 'low',
                'evidence': '[Parsing Failure] The model output did not contain this facet or failed schema validation.'
            })
            
    return final_results

def score_observable_facets(convo_text: str, candidates: list[dict]) -> list[dict]:
    """
    Orchestrates the prompt generation, model invocation, output cleaning, 
    parsing, and validation for observable facets.
    """
    if not candidates:
        return []
        
    messages = generate_prompt(convo_text, candidates)
    
    try:
        raw_output = request_llm_scoring(messages)
    except Exception as e:
        # Graceful API failure handling: return fallback abstentions for all candidates
        print(f"[API Execution Failure]: {e}. Returning safe invalid_model_output defaults.")
        return [
            {
                'facet': c['normalized_facet'],
                'status': 'invalid_model_output',
                'score': None,
                'confidence': 'low',
                'evidence': f'[API Failure] Connection error: {str(e)}'
            }
            for c in candidates
        ]
        
    cleaned_output = clean_model_output(raw_output)
    
    # Attempt strict JSON loads
    parsed_list = []
    try:
        parsed_list = json.loads(cleaned_output)
        if not isinstance(parsed_list, list):
            parsed_list = parse_with_regex(cleaned_output)
    except Exception:
        # Regex recovery
        parsed_list = parse_with_regex(cleaned_output)
        
    return validate_and_standardize_results(parsed_list, candidates)
