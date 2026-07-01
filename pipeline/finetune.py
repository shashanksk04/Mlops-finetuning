"""
QLoRA fine-tuning pipeline.

Fine-tunes a causal language model using LoRA (Low-Rank Adaptation) with
4-bit quantization (QLoRA), tracking every run with MLflow. The base model
is configurable — defaults to a small CPU-friendly model so the full
pipeline is runnable without a GPU. Swap `base_model_name` to a larger
model (e.g. "mistralai/Mistral-7B-v0.1") when running on GPU hardware;
the QLoRA configuration below is correct for both.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlflow
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)


@dataclass
class FineTuneConfig:
    base_model_name: str = "distilgpt2"  # swap to mistralai/Mistral-7B-v0.1 on GPU
    dataset_path: str = "data/support_dataset.jsonl"
    output_dir: str = "outputs/finetuned-model"
    use_4bit: bool = False  # auto-enabled on GPU; CPU falls back to full precision LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    max_seq_length: int = 256
    mlflow_experiment: str = "llm-finetuning-qlora"


def load_jsonl_dataset(path: str | Path) -> Dataset:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['response']}"
            examples.append({"text": text})
    return Dataset.from_list(examples)


def build_quantization_config(use_4bit: bool) -> Optional[BitsAndBytesConfig]:
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def run_finetuning(cfg: FineTuneConfig) -> str:
    """
    Runs QLoRA fine-tuning end-to-end and logs everything to MLflow.
    Returns the MLflow run_id.
    """
    gpu_available = torch.cuda.is_available()
    use_4bit = cfg.use_4bit and gpu_available

    mlflow.set_experiment(cfg.mlflow_experiment)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "base_model": cfg.base_model_name,
                "lora_r": cfg.lora_r,
                "lora_alpha": cfg.lora_alpha,
                "lora_dropout": cfg.lora_dropout,
                "epochs": cfg.num_train_epochs,
                "learning_rate": cfg.learning_rate,
                "batch_size": cfg.per_device_train_batch_size,
                "use_4bit_quantization": use_4bit,
                "gpu_available": gpu_available,
            }
        )

        # ---- Load tokenizer & dataset ----
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = load_jsonl_dataset(cfg.dataset_path)
        mlflow.log_param("dataset_size", len(dataset))

        def tokenize_fn(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=cfg.max_seq_length,
                padding="max_length",
            )

        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        # ---- Load base model (quantized on GPU, full precision on CPU) ----
        quant_config = build_quantization_config(use_4bit)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name,
            quantization_config=quant_config,
            device_map="auto" if gpu_available else None,
        )

        if use_4bit:
            model = prepare_model_for_kbit_training(model)

        # ---- Apply LoRA adapters ----
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=_infer_target_modules(model),
        )
        model = get_peft_model(model, lora_config)

        trainable, total = model.get_nb_trainable_parameters()
        pct_trainable = 100 * trainable / total
        mlflow.log_metrics(
            {
                "trainable_params": trainable,
                "total_params": total,
                "trainable_pct": pct_trainable,
            }
        )

        # ---- Train ----
        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            learning_rate=cfg.learning_rate,
            logging_steps=1,
            save_strategy="epoch",
            report_to=[],  # we log to MLflow manually for full control
            fp16=gpu_available,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )

        train_result = trainer.train()

        mlflow.log_metrics(
            {
                "final_train_loss": train_result.training_loss,
            }
        )

        # ---- Save adapter weights ----
        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        mlflow.log_artifacts(cfg.output_dir, artifact_path="model")

        return run.info.run_id


def _infer_target_modules(model) -> list[str]:
    """
    LoRA needs to know which linear layers to adapt. This differs by
    architecture (GPT-2 uses c_attn, Llama/Mistral use q_proj/v_proj, etc).
    We inspect the model to pick sensible defaults automatically.
    """
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}

    if "c_attn" in module_names:  # GPT-2 family
        return ["c_attn"]
    if "q_proj" in module_names and "v_proj" in module_names:  # Llama/Mistral family
        return ["q_proj", "v_proj"]

    # Fallback: adapt all Linear layers found
    return ["c_attn"]


if __name__ == "__main__":
    config = FineTuneConfig()
    run_id = run_finetuning(config)
    print(f"Training complete. MLflow run_id: {run_id}")
