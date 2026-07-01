"""
FastAPI inference server for the fine-tuned model.

Run with:  uvicorn pipeline.serve:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", "outputs/finetuned-model")
BASE_MODEL = os.environ.get("BASE_MODEL", "distilgpt2")

model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if Path(MODEL_DIR).exists():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        model = PeftModel.from_pretrained(base_model, MODEL_DIR)
        model.eval()
        model_state["tokenizer"] = tokenizer
        model_state["model"] = model
    else:
        model_state["tokenizer"] = None
        model_state["model"] = None
    yield
    model_state.clear()


app = FastAPI(title="Fine-Tuned Support Model API", lifespan=lifespan)


class GenerateRequest(BaseModel):
    instruction: str
    max_new_tokens: int = 80


class GenerateResponse(BaseModel):
    response: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_state.get("model") is not None,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    model = model_state.get("model")
    tokenizer = model_state.get("tokenizer")

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Expected fine-tuned weights at '{MODEL_DIR}'. "
            f"Run the training pipeline first.",
        )

    prompt = f"### Instruction:\n{req.instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response_text = full_text.split("### Response:\n")[-1].strip()

    return GenerateResponse(response=response_text)
