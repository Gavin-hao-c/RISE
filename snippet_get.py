"""
Call a local Qwen model to iteratively generate fix-code snippets for common CWE vulnerabilities.

Workflow:
1. Prompt "List 5 common CWE vulnerability types" to get 5 CWE types from the model.
2. Prompt "Generate the common fix code for these 5 vulnerabilities" to get remediation code.
3. Each round, feed the model's previous reply and the prompt back in to continue generating.
4. Filter the generated snippets: keep only those with fewer than 10 lines and not already in snippet_pool.
5. Append valid snippets to snippet_pool (saved as a JSON file) until the pool reaches 30.
"""

import os
import re
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

# ============ Configurable parameters ============
MODEL_PATH = "Qwen/Qwen3.5-4B"   # Local model path or name, change as needed
POOL_FILE = "snippet_pool.json"  # Snippet pool storage file
TARGET_COUNT = 30                # Target number of snippets
MAX_LINES = 10                   # Line-count limit (keep snippets with fewer than this)
MAX_ROUNDS = 50                  # Safety cap to avoid an infinite loop from too many duplicates
GEN_PROMPT = "Generate the common fix code for these 5 vulnerabilities"
CWE_PROMPT = "List 5 common CWE vulnerability types"

# ============ Load the model ============
print(f"Loading model: {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
print("Model loaded.")


def chat(messages, max_new_tokens=1024):
    """Conversational inference: take a messages list, return the text generated this turn."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    # Keep only the newly generated part
    output_ids = generated[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def load_pool():
    """Load the existing snippet pool; return an empty list if it does not exist."""
    if os.path.exists(POOL_FILE):
        with open(POOL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("snippets", [])
    return []


def save_pool(snippets):
    """Write the snippet pool back to the JSON file."""
    with open(POOL_FILE, "w", encoding="utf-8") as f:
        json.dump({"snippets": snippets}, f, ensure_ascii=False, indent=2)


def extract_code_blocks(text):
    """Extract all ``` fenced code blocks from the model reply."""
    blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]


def normalize(snippet):
    """Normalize a snippet for dedup comparison (strip trailing whitespace and leading/trailing blank lines)."""
    return "\n".join(line.rstrip() for line in snippet.strip().splitlines())


def is_valid(snippet, pool):
    """Validate a snippet: fewer than MAX_LINES lines and not a duplicate of any pool entry."""
    line_count = len(snippet.splitlines())
    if line_count >= MAX_LINES:
        return False
    existing = {normalize(s) for s in pool}
    return normalize(snippet) not in existing


def main():
    pool = load_pool()
    print(f"Snippet pool currently holds {len(pool)} snippets.")

    # Step 1: get 5 common CWE vulnerability types
    cwe_reply = chat([{"role": "user", "content": CWE_PROMPT}])
    print("=" * 60)
    print("The 5 CWE types returned by the model:")
    print(cwe_reply)
    print("=" * 60)

    # Step 2: build the conversation context from the reply above, then loop to generate fix code
    messages = [
        {"role": "user", "content": CWE_PROMPT},
        {"role": "assistant", "content": cwe_reply},
        {"role": "user", "content": GEN_PROMPT},
    ]

    rounds = 0
    while len(pool) < TARGET_COUNT and rounds < MAX_ROUNDS:
        rounds += 1
        reply = chat(messages)

        # Feed this reply + the prompt back into the model to continue generating
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": GEN_PROMPT})

        # Extract and filter code blocks
        blocks = extract_code_blocks(reply)
        added = 0
        for block in blocks:
            if is_valid(block, pool):
                pool.append(block)
                added += 1
                if len(pool) >= TARGET_COUNT:
                    break

        save_pool(pool)
        print(
            f"[Round {rounds}] extracted {len(blocks)} code blocks, "
            f"added {added} valid snippets, pool now has {len(pool)}."
        )

    if len(pool) >= TARGET_COUNT:
        print(f"\nDone! snippet_pool collected {len(pool)} snippets, saved to {POOL_FILE}.")
    else:
        print(
            f"\nReached max rounds {MAX_ROUNDS}, stopping. Currently {len(pool)} snippets "
            f"(a high duplicate rate may prevent reaching {TARGET_COUNT}; "
            f"try raising temperature or diversifying the prompt)."
        )


if __name__ == "__main__":
    main()
