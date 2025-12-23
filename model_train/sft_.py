from pathlib import Path
from dataset_ import *
import json
from torch.utils.data import DataLoader

CASE = 1

# This is the base directory for the project.
BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR / "data/train" / train_data_list[0]) as json_file:
    train_data = json.load(json_file)

if CASE == 1:
    with open(BASE_DIR / "data/test" / test_data_list[0]) as json_file:
        test_data = json.load(json_file)
if CASE == 2:
    with open(BASE_DIR / "data/test" / test_data_list[1]) as json_file:
        test_data = json.load(json_file)
if CASE == 3:
    with open(BASE_DIR / "data/test" / test_data_list[2]) as json_file:
        test_data = json.load(json_file)

# 这是一个量化版的模型。所以在微调的时候要使用QLoRA。
# 需要添加梯度积累步数，以弥补批量大小过小的问题。
model = "unsloth/Qwen2.5-7B-unsloth-bnb-4bit"

train_dataset = SFTDataset(train_data, tokenizer)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, tokenizer),
)

test_dataset = SFTDataset(test_data, tokenizer)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda batch: collate_fn(batch, tokenizer),
)



