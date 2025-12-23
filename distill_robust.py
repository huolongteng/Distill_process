from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.gkd import GKDConfig, GKDTrainer
import torch

# Step 1: 准备学生和教师的模型与分词器
student_model_id = "Qwen/Qwen2-0.5B-Instruct"
teacher_model_id = "Qwen/Qwen2-1.5B-Instruct"
student_tokenizer = AutoTokenizer.from_pretrained(student_model_id)
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
# Step 2: 确保分词器有pad_token，避免批处理时报错
if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = student_tokenizer.eos_token
if teacher_tokenizer.pad_token is None:
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

# Step 3: 加载模型
student_model = AutoModelForCausalLM.from_pretrained(student_model_id)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_id)

# Step 4: 将学生模型的embedding和lm_head扩到教师词表大小
teacher_vocab_size = teacher_model.config.vocab_size
old_vocab_size = student_model.get_input_embeddings().weight.shape[0]
if old_vocab_size < teacher_vocab_size:
    student_model.resize_token_embeddings(teacher_vocab_size)
    # 如果输出头未绑定，手动扩展权重
    output_layer = student_model.get_output_embeddings()
    if output_layer.weight.shape[0] != teacher_vocab_size:
        new_output_layer = torch.nn.Linear(output_layer.in_features, teacher_vocab_size, bias=False)
        with torch.no_grad():
            new_output_layer.weight[: output_layer.weight.shape[0]] = output_layer.weight
        student_model.set_output_embeddings(new_output_layer)

# Step 5: 随机提取1000条数据并划分训练/验证
raw_dataset = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train[:10000]")
sampled_dataset = raw_dataset.shuffle(seed=42).select(range(1000))
train_dataset = sampled_dataset.select(range(800))
eval_dataset = sampled_dataset.select(range(800, 1000))

# Step 6: 将数据集格式化为messages结构
column_names = train_dataset.column_names

def _format_to_messages(example):
    # 尽量从常见字段获取内容
    content = example.get("text") or example.get("prompt") or ""
    assistant_hint = example.get("response") or example.get("output") or "请给出详细的解答。"
    return {
        "messages": [
            {"role": "user", "content": str(content)},
            {"role": "assistant", "content": str(assistant_hint)},
        ]
    }

train_dataset = train_dataset.map(_format_to_messages, remove_columns=column_names)
eval_dataset = eval_dataset.map(_format_to_messages, remove_columns=column_names)

# Step 7: 配置并运行GKD训练
training_args = GKDConfig(output_dir="gkd-model",
                          per_device_train_batch_size=8,
                          num_train_epochs=1,
)
trainer = GKDTrainer(
    model=student_model,
    teacher_model=teacher_model,
    args=training_args,
    processing_class=student_tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()

# Step 8: 保存训练好的学生模型
trainer.save_model("gkd-trained-student")
