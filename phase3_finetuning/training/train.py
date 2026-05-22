from unsloth import FastLanguageModel
from datasets import load_from_disk
from trl import SFTTrainer,SFTConfig
from unsloth import train_on_responses_only


dataset = load_from_disk("/home/charan/projects/llm-fine-tuning/phase3_finetuning/data/processed/train")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    max_seq_length = 1024,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)


    
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 1024,

    args = SFTConfig(
        max_length=None,
        padding_free=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir = "phase3_finetuning/models/outputs",
    ),
)
train_on_responses_only(trainer=trainer, instruction_part = "<|user|>", response_part = "<|assistant|>")
trainer.train()
model.save_pretrained("/home/charan/projects/llm-fine-tuning/phase3_finetuning/models/phi3_lora")
tokenizer.save_pretrained("/home/charan/projects/llm-fine-tuning/phase3_finetuning/models/phi3_lora")
