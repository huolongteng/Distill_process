from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.gold import GOLDConfig, GOLDTrainer

# Step 1: pick tokenizers separately so the student keeps its own vocab/embedding size
# （步骤1：分别加载教师和学生的tokenizer，避免学生的embedding尺寸被改动）
student_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
teacher_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-PRM-7B")

# Step 2: load teacher and student models
# （步骤2：加载教师模型和学生模型）
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
teacher_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-PRM-7B")

# Step 3: pull the OpenR1-Math dataset and shuffle it
# （步骤3：加载并乱序OpenR1-Math数据集）
raw_dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train")
shuffled_dataset = raw_dataset.shuffle(seed=42)

# Step 4: cut out 2000 samples for training and 200 for evaluation
# （步骤4：切分2000条训练数据和200条测试数据）
train_raw = shuffled_dataset.select(range(2000))
eval_raw = shuffled_dataset.select(range(2000, 2200))

# Step 5: map every sample into chat messages that GOLD expects
# （步骤5：把样本整理成GOLD需要的messages格式）
def to_messages(example):
    # 优先使用已经存在的messages字段
    if "messages" in example and example["messages"]:
        return {"messages": example["messages"]}

    # 其次尝试常见的问答键名
    question = example.get("question") or example.get("problem") or example.get("input") or example.get("prompt")
    answer = example.get("answer") or example.get("solution") or example.get("response") or example.get("output")

    # 兜底：没有字段就把整个样本转成字符串
    if question is None:
        question = str(example)
    if answer is None:
        answer = ""

    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }

train_dataset = train_raw.map(to_messages, remove_columns=train_raw.column_names)
eval_dataset = eval_raw.map(to_messages, remove_columns=eval_raw.column_names)

# Step 6: set up trainer configuration (enable ULD for tokenizer/vocab mismatch)
training_args = GOLDConfig(
    output_dir="gkd-model",
    per_device_train_batch_size=1,
    use_uld_loss=True,
    teacher_tokenizer_name_or_path="Qwen/Qwen2.5-Math-PRM-7B"
)

trainer = GOLDTrainer(
    model=model,
    teacher_model=teacher_model,
    args=training_args,
    processing_class=student_tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


# Step 7: train and then save the student weights
# （步骤7：开始训练，并在结束后保存学生模型权重）
trainer.train()
model.save_pretrained("student-model")
student_tokenizer.save_pretrained("student-model")
