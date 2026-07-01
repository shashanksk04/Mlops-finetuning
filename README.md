# 🔧 MLOps Pipeline for LLM Fine-Tuning (QLoRA + MLflow)

An end-to-end MLOps pipeline that fine-tunes a language model on a custom
instruction dataset using **LoRA / QLoRA**, tracks every run with
**MLflow**, and gates model promotion behind an automated **evaluation
check** — the same pre-production quality-gate pattern used in CI/CD
pipelines, applied here to model artifacts.

The pipeline is fully runnable on CPU using `distilgpt2` (so anyone can
clone and verify it end-to-end without a GPU), and is written to drop in
a larger model (e.g. `mistralai/Mistral-7B-v0.1`) with 4-bit QLoRA
quantization simply by changing the config — no code changes required.

## Architecture

```
 ┌──────────────┐     ┌─────────────────┐     ┌────────────────┐     ┌───────────────┐
 │  Dataset      │ --> │  LoRA / QLoRA    │ --> │  MLflow         │ --> │  Evaluation    │
 │  (.jsonl)     │     │  Fine-Tuning     │     │  Run Tracking   │     │  Gate          │
 └──────────────┘     └─────────────────┘     └────────────────┘     └───────┬────────┘
                                                                                │
                                              ┌─────────────────────────────────┴────┐
                                              │  PASS → tag "production-ready"        │
                                              │  FAIL → tag "rejected", block deploy   │
                                              └─────────────────────────────────┬────┘
                                                                                │
                                                                                v
                                                                  ┌──────────────────────┐
                                                                  │  FastAPI Serving      │
                                                                  │  (Docker / GKE-ready) │
                                                                  └──────────────────────┘
```

## How it works

1. **`pipeline/prepare_dataset.py`** — builds a small instruction-tuning dataset (customer-support domain) in `.jsonl` format. Swap this for a data warehouse / S3 pull in production.
2. **`pipeline/finetune.py`** — loads the base model, applies a LoRA adapter (4-bit QLoRA quantization auto-enables on GPU, falls back to full-precision LoRA on CPU), trains, and logs every parameter, metric, and the resulting adapter weights to MLflow.
3. **`pipeline/evaluate.py`** — loads the fine-tuned adapter, computes perplexity on a held-out prompt set, and only tags the MLflow run `production-ready` if it beats the configured threshold. Failing runs are tagged `rejected` and the gate exits non-zero, which fails the CI job — exactly how a bad build fails a deployment.
4. **`pipeline/serve.py`** — a FastAPI server that loads the production-tagged adapter and serves `/generate` for inference, with a `/health` endpoint for k8s liveness probes.

## Quickstart

### 1. Install

```bash
git clone <this-repo>
cd mlops-finetuning
pip install -r requirements.txt
```

### 2. Prepare the dataset

```bash
python -m pipeline.prepare_dataset
```

### 3. Fine-tune (CPU-friendly, ~1-2 minutes for the default 3 epochs on distilgpt2)

```bash
python -m pipeline.finetune
```

This prints an MLflow `run_id` at the end — copy it.

### 4. Run the evaluation gate

```bash
python -m pipeline.evaluate <run_id> outputs/finetuned-model
```

Exits `0` and prints `PASSED` if the model clears the perplexity threshold; exits `1` and prints `FAILED` otherwise — this is the exact command your CI/CD would run before allowing a deploy.

### 5. View experiment tracking in the MLflow UI

```bash
mlflow ui
```

Open `http://localhost:5000` to see every run, its hyperparameters, training loss curve, and evaluation perplexity — fully comparable across runs.

### 6. Serve the fine-tuned model

```bash
uvicorn pipeline.serve:app --reload
```

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Customer says: I want a refund for my last order."}'
```

### 7. Docker

```bash
docker build -t mlops-finetuning .
docker run -p 8000:8000 -v $(pwd)/outputs:/app/outputs mlops-finetuning
```

## Running tests

```bash
# Fast tests only (no training, runs in seconds)
pytest tests/ -v -m "not slow"

# Full smoke test — actually trains distilgpt2 for 1 epoch and runs the eval gate (~30-60s on CPU)
pytest tests/ -v -m "slow"
```

The CI pipeline (`.github/workflows/ci.yml`) runs both: fast tests on every push, and the full training + evaluation smoke test in a separate job, with the resulting MLflow run uploaded as a build artifact for inspection.

## Going to a real 7B model on GPU

Change one line in `pipeline/finetune.py`:

```python
base_model_name: str = "mistralai/Mistral-7B-v0.1"
use_4bit: bool = True
```

The `BitsAndBytesConfig` (4-bit NF4 quantization, double quantization, float16 compute) and LoRA target-module auto-detection already handle the Llama/Mistral architecture — no other code changes needed. This is the exact configuration that, on real 7B-scale models, cuts GPU memory usage by roughly 75% versus full fine-tuning while staying within a few points of full fine-tuning accuracy.

## Project structure

```
mlops-finetuning/
├── pipeline/
│   ├── prepare_dataset.py   # Dataset generation
│   ├── finetune.py          # LoRA/QLoRA training + MLflow logging
│   ├── evaluate.py          # Perplexity-based evaluation gate
│   └── serve.py             # FastAPI inference server
├── tests/
│   └── test_pipeline.py     # Fast tests + full training smoke test
├── data/                    # Generated dataset lands here
├── Dockerfile
├── .github/workflows/ci.yml # Test -> train+evaluate -> docker build
└── requirements.txt
```

## Design decisions

- **CPU-runnable by default** — using `distilgpt2` means anyone (recruiter included) can clone this and see real training happen in under a minute, with no GPU or cloud bill required. The QLoRA code path is real and switches on automatically when a GPU is detected.
- **Evaluation gate blocks promotion, not just reports a number** — `gate_and_promote()` actually tags MLflow runs `production-ready` or `rejected` and the CLI exits non-zero on failure, so it can wire directly into a CI/CD `if` check the same way a failing test suite blocks a deploy.
- **Auto-detected LoRA target modules** — different model architectures expose different attention layer names (GPT-2's `c_attn` vs Llama/Mistral's `q_proj`/`v_proj`); `_infer_target_modules()` inspects the loaded model so the same training script works across architectures without manual config.

## Tech stack

`Python` · `HuggingFace Transformers` · `PEFT (LoRA/QLoRA)` · `bitsandbytes` · `MLflow` · `FastAPI` · `Docker` · `GitHub Actions`

## License

MIT
