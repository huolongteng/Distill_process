import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

DATASET_ID = "math-ai/aime24"
SPLIT = "test"

TEACHER_MODEL = "open-thoughts/OpenThinker3-7B"
STUDENT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 128

SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "Solve the math problem. Output ONLY the final answer as an integer between 0 and 999.\n"
    "Do not include any words, explanation, LaTeX, or punctuation—only the integer."
)


@torch.inference_mode()
def predict_answer(model, tokenizer, problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Deterministic decoding
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


def load_model_and_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    mdl.eval()
    return mdl, tok


def main():
    ds = load_dataset(DATASET_ID, split=SPLIT)

    all_results = []
    for ex in ds:
        all_results.append({
            "id": ex["id"],
            "problem": ex["problem"],
            "solution": ex["solution"]
        })

    # Load teacher model
    teacher_model, teacher_tok = load_model_and_tokenizer(TEACHER_MODEL)

    for result in tqdm(all_results, desc="Teacher predictions"):
        raw = predict_answer(teacher_model, teacher_tok, result["problem"])
        result["teacher_raw"] = raw

    # Free teacher
    del teacher_model
    torch.cuda.empty_cache()

    # Load student model
    student_model, student_tok = load_model_and_tokenizer(STUDENT_MODEL)

    for result in tqdm(all_results, desc="Student predictions"):
        raw = predict_answer(student_model, student_tok, result["problem"])
        result["student_raw"] = raw

    del student_model
    torch.cuda.empty_cache()

    # Save to file for manual check
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("Results saved to results.json for manual inspection.")

    # Print first 3 as example
    print("\nFirst 3 results:")
    for r in all_results[:3]:
        print(f"ID: {r['id']}")
        print(f"Problem: {r['problem'][:100]}...")
        print(f"Solution: {r['solution']}")
        print(f"Teacher Raw: {r['teacher_raw']}")
        print(f"Student Raw: {r['student_raw']}")
        print("---")


if __name__ == "__main__":
    main()
