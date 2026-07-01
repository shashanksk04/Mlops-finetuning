"""
Evaluation gate.

After fine-tuning, this module evaluates the model against a held-out
set of prompts and computes perplexity. If the model fails to meet the
quality threshold, it is NOT promoted — this mirrors the pre-production
quality gates used in CI/CD pipelines, applied here to model artifacts
instead of application code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlflow
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


@dataclass
class EvalResult:
    perplexity: float
    passed: bool
    threshold: float


EVAL_PROMPTS = [
    "### Instruction:\nCustomer says: 'I want a refund for my last order.'\n\n### Response:\n",
    "### Instruction:\nCustomer says: 'How long does shipping take?'\n\n### Response:\n",
    "### Instruction:\nCustomer says: 'My account got locked.'\n\n### Response:\n",
]


def compute_perplexity(model, tokenizer, prompts: list[str]) -> float:
    model.eval()
    losses = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())

    avg_loss = sum(losses) / len(losses)
    return math.exp(avg_loss)


def evaluate_model(
    base_model_name: str,
    adapter_path: str,
    perplexity_threshold: float = 150.0,
) -> EvalResult:
    """
    Loads the base model + LoRA adapter, computes perplexity on held-out
    prompts, and decides pass/fail against `perplexity_threshold`.

    A real production threshold would be tuned against your specific
    domain and base model; this default is calibrated for the small
    CPU-friendly model used in this repo's quickstart.
    """
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)

    ppl = compute_perplexity(model, tokenizer, EVAL_PROMPTS)
    passed = ppl <= perplexity_threshold

    mlflow.log_metrics({"eval_perplexity": ppl, "eval_passed": int(passed)})

    return EvalResult(perplexity=ppl, passed=passed, threshold=perplexity_threshold)


def gate_and_promote(run_id: str, adapter_path: str, base_model_name: str) -> bool:
    """
    Runs the evaluation gate against a given MLflow run. If the model
    passes, it is tagged as 'production-ready' in MLflow. If it fails,
    the run is tagged as 'rejected' and the pipeline should halt
    deployment (this is what the CI step calls).
    """
    with mlflow.start_run(run_id=run_id):
        result = evaluate_model(base_model_name, adapter_path)

        if result.passed:
            mlflow.set_tag("promotion_status", "production-ready")
        else:
            mlflow.set_tag("promotion_status", "rejected")

    return result.passed


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m pipeline.evaluate <run_id> <adapter_path>")
        sys.exit(1)

    run_id = sys.argv[1]
    adapter_path = sys.argv[2]

    passed = gate_and_promote(run_id, adapter_path, base_model_name="distilgpt2")
    print(f"Evaluation gate {'PASSED' if passed else 'FAILED'}")
    sys.exit(0 if passed else 1)
