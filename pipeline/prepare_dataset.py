"""
Dataset preparation for fine-tuning.

Generates a small instruction-following dataset (domain: customer support
responses) and saves it in a format ready for tokenization. In a real
production setting this would pull from a labeled dataset in S3/a data
warehouse — here we ship a small synthetic dataset so the pipeline is
runnable end-to-end without external data dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict


def build_support_dataset() -> List[Dict[str, str]]:
    """A small domain-specific instruction dataset: customer support replies."""
    examples = [
        {
            "instruction": "Customer says: 'My order hasn't arrived after 10 days.'",
            "response": "I'm sorry for the delay. Let me look up your order and provide an updated delivery estimate right away.",
        },
        {
            "instruction": "Customer says: 'I want to cancel my subscription.'",
            "response": "I can help with that. I'll process the cancellation now — you won't be charged again, and your access continues until the end of the current billing period.",
        },
        {
            "instruction": "Customer says: 'The product I received is damaged.'",
            "response": "I'm sorry to hear that. Please share a photo of the damage and I'll arrange a free replacement or full refund immediately.",
        },
        {
            "instruction": "Customer says: 'Can I change my shipping address after ordering?'",
            "response": "If the order hasn't shipped yet, I can update the address for you right now. Let me check the current status of your order.",
        },
        {
            "instruction": "Customer says: 'Why was I charged twice this month?'",
            "response": "I apologize for the confusion. Let me check your billing history and refund any duplicate charge within the next 2 business days.",
        },
        {
            "instruction": "Customer says: 'How do I reset my password?'",
            "response": "You can reset your password by clicking 'Forgot Password' on the login page. I'll also send a reset link to your registered email now.",
        },
        {
            "instruction": "Customer says: 'Is there a student discount available?'",
            "response": "Yes! We offer a 15% student discount. Please verify your student status through our partner portal and the discount will apply automatically.",
        },
        {
            "instruction": "Customer says: 'My coupon code isn't working.'",
            "response": "I'm sorry about that. Let me check the coupon code and apply the discount manually to your order if it's valid but not applying correctly.",
        },
        {
            "instruction": "Customer says: 'Can I get an invoice for my purchase?'",
            "response": "Of course! I'll generate a detailed invoice and send it to your registered email within the next few minutes.",
        },
        {
            "instruction": "Customer says: 'The app keeps crashing on my phone.'",
            "response": "Sorry for the trouble. Could you let me know your phone model and app version? In the meantime, try reinstalling the app — that resolves most crash issues.",
        },
        {
            "instruction": "Customer says: 'I never received the confirmation email.'",
            "response": "I'll resend the confirmation email right now. Please also check your spam folder in case it was filtered there.",
        },
        {
            "instruction": "Customer says: 'Do you offer international shipping?'",
            "response": "Yes, we ship to over 40 countries. Shipping costs and delivery times vary by location — I can give you an exact estimate if you share your country.",
        },
    ]
    return examples


def save_dataset(output_path: str | Path) -> int:
    examples = build_support_dataset()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    return len(examples)


if __name__ == "__main__":
    n = save_dataset("data/support_dataset.jsonl")
    print(f"Saved {n} examples to data/support_dataset.jsonl")
