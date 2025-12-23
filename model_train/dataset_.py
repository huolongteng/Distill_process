import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

train_data_list = [
    "Instruction Dataset of LogLM.json"
]

test_data_list = [
    "log_interpretation_test.json",
    "root_cause_analysis_test.json",
    "solution_recommendation_test.json"
]

def collate_fn(batch, tokenizer):
    input_ids = [x["input_ids"] for x in batch]
    labels = [x["labels"] for x in batch]

    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100,
    )

    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

def build_prompt(example):
    if example["input"].strip():
        return (
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Input:\n"
            f"{example['input']}\n\n"
            "### Response:\n"
        )
    else:
        return (
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Response:\n"
        )

class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        prompt = build_prompt(example)
        answer = example["output"]

        # prompt + answer 拼在一起喂给 causal LM
        full_text = prompt + answer

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]

        # labels：prompt 部分 mask 掉
        labels = input_ids.copy()

        prompt_len = len(
            self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )["input_ids"]
        )

        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

