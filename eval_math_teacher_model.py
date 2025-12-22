from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import re
from tqdm.auto import tqdm

# 1. Load the dataset
dataset = load_dataset("openai/gsm8k", "main")
# We'll use the test split for evaluation
# For a quick test, you might want to slice it, e.g., dataset['test'][:10]
test_data = dataset['test']

# 2. Load the model and tokenizer
model_name = "wooferclaw/Qwen2.5-1.5B-Open-R1-Distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Use left padding for decoder-only models to avoid generation warnings and speed issues
tokenizer.padding_side = "left"
# Ensure a pad token is available; fall back to EOS if the tokenizer lacks one
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto", # Automatically map model to available devices (GPU if available)
    torch_dtype=torch.bfloat16 # Use bfloat16 for potentially faster inference and less memory
)

# Keep the model in eval mode and allow faster matmul on GPU
model.eval()
torch.backends.cuda.matmul.allow_tf32 = True

# 3. Define a function to extract the answer from the model's output and the ground truth
def extract_answer(text):
    # GSM8k answers are typically formatted as '#### <number>'
    # Model outputs might vary, so we try to find the last number in the text
    # First, try to find the '#### ' pattern for ground truth
    match_gt = re.search(r'#### (\-?\d+\.?\d*)', text)
    if match_gt: # For ground truth, this should always work
        return float(match_gt.group(1))

    # For model output, look for the last number, often after 'The answer is' or similar phrasing
    match_model = re.findall(r'\-?\d+\.?\d*', text)
    if match_model:
        try:
            return float(match_model[-1])
        except ValueError:
            return None # Could not convert to float
    return None

# 4. Prepare prompts and run inference
correct_predictions = 0
total_samples = len(test_data)

# Set generation parameters
gen_kwargs = {
    "max_new_tokens": 256,
    "do_sample": False,
    "num_beams": 1,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    "use_cache": True, # speed up decoding
}

print(f"Starting evaluation on {total_samples} samples...")

# Process in batches for efficiency if the dataset is large
batch_size = 64 # Adjust based on your GPU memory
predictions = []

for i in tqdm(range(0, total_samples, batch_size), desc="Evaluating"): # Use tqdm for progress bar
    # Slice questions/answers directly to avoid dataset overhead
    end = min(i + batch_size, total_samples)
    batch_questions = test_data['question'][i:end]
    batch_answers_gt = test_data['answer'][i:end]

    # Format the prompt for the model (adjust if model requires a specific chat format)
    # For open-ended models, a simple prompt usually works.
    prompts = [f"Question: {q}\nAnswer:" for q in batch_questions]

    # Tokenize once per batch and generate in inference mode for speed
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    for j, generated_text in enumerate(generated_texts):
        # Use the question from the batch for consistency, although not strictly needed for accuracy calculation
        question = batch_questions[j]
        ground_truth_answer_full = batch_answers_gt[j]

        # Extract numerical answers
        predicted_value = extract_answer(generated_text)
        true_value = extract_answer(ground_truth_answer_full)

        if predicted_value is not None and true_value is not None:
            if abs(predicted_value - true_value) < 1e-6: # Account for floating point inaccuracies
                correct_predictions += 1
        # else: # You might want to log cases where extraction fails
            # print(f"Warning: Could not extract answer for sample {i+j}. Model output: {generated_text[:100]}...")
            # print(f"Ground Truth: {ground_truth_answer_full}")

accuracy = (correct_predictions / total_samples) * 100

print(f"\nEvaluation Complete!")
print(f"Total samples: {total_samples}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")
