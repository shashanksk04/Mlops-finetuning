"""
Unit tests for the MLOps fine-tuning pipeline.

`test_full_pipeline_smoke_test` actually runs a tiny real training loop
(1 epoch, distilgpt2, 12 examples) — this takes ~30-60s on CPU and proves
the QLoRA/LoRA + MLflow + evaluation gate flow genuinely works end-to-end,
not just that the code parses.
"""

import json
import os
from pathlib import Path

import pytest

from pipeline.prepare_dataset import build_support_dataset, save_dataset
from pipeline.finetune import FineTuneConfig, load_jsonl_dataset, run_finetuning
from pipeline.evaluate import evaluate_model, EVAL_PROMPTS

DATA_DIR = Path(__file__).parent.parent / "data"


def test_build_support_dataset_returns_examples():
    examples = build_support_dataset()
    assert len(examples) >= 10
    for ex in examples:
        assert "instruction" in ex
        assert "response" in ex
        assert len(ex["instruction"]) > 0
        assert len(ex["response"]) > 0


def test_save_dataset_writes_valid_jsonl(tmp_path):
    out_path = tmp_path / "dataset.jsonl"
    n = save_dataset(out_path)

    assert out_path.exists()
    assert n > 0

    with open(out_path) as f:
        lines = f.readlines()
    assert len(lines) == n

    for line in lines:
        row = json.loads(line)
        assert "instruction" in row and "response" in row


def test_load_jsonl_dataset_formats_text_correctly(tmp_path):
    save_dataset(tmp_path / "dataset.jsonl")
    dataset = load_jsonl_dataset(tmp_path / "dataset.jsonl")

    assert len(dataset) > 0
    sample = dataset[0]["text"]
    assert "### Instruction:" in sample
    assert "### Response:" in sample


@pytest.mark.slow
def test_full_pipeline_smoke_test(tmp_path):
    """
    Real end-to-end smoke test: trains distilgpt2 with LoRA for 1 epoch
    on the sample dataset, then runs the evaluation gate against it.
    Marked 'slow' so it can be excluded from fast local iteration with
    `pytest -m "not slow"`, but it DOES run in CI.
    """
    dataset_path = tmp_path / "support_dataset.jsonl"
    save_dataset(dataset_path)

    output_dir = tmp_path / "finetuned-model"

    cfg = FineTuneConfig(
        base_model_name="distilgpt2",
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        mlflow_experiment="test-experiment",
    )

    run_id = run_finetuning(cfg)
    assert run_id is not None
    assert output_dir.exists()
    assert (output_dir / "adapter_config.json").exists()

    result = evaluate_model(
        base_model_name="distilgpt2",
        adapter_path=str(output_dir),
        perplexity_threshold=1e6,  # generous threshold, just confirming the gate runs
    )
    assert result.perplexity > 0
    assert result.passed is True
