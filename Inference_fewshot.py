#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Better VQA answers — NO TRAINING REQUIRED.

Uses your existing checkpoint-100 adapter as-is. Instead of retraining
(risky given today's crashes), this shows the model 2-3 examples of the
answer style you want directly in the prompt (few-shot prompting). The
model imitates the pattern immediately, at inference time, with zero GPU
training risk.

Run:
    python inference_fewshot.py --image path/to/image.png --question "How many cars are visible?"

Or interactively:
    python inference_fewshot.py --interactive
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

ROOT_DIR = Path(__file__).resolve().parent
MODEL_NAME = str(ROOT_DIR / "model")
ADAPTER_DIR = ROOT_DIR / "qwen2_5_vl_vrsbench_lora" / "checkpoint-100"  # your existing, already-trained adapter

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 384 * 28 * 28

# Text-only few-shot examples: shown to the model as prior conversation turns
# so it learns the answer style by imitation, without needing any images for
# these examples (kept lightweight and fast).
FEWSHOT_EXAMPLES = [
    {
        "question": "How many cars are visible?",
        "answer": "There are 3 cars visible, parked along the right side of the lane near a small building.",
    },
    {
        "question": "Is there a building near the road?",
        "answer": "Yes, there is a building near the road, visible with a flat roof close to the intersection.",
    },
    {
        "question": "What is the dominant land cover?",
        "answer": "The dominant land cover is grassland, covering most of the image with scattered patches of bare soil.",
    },
]

SYSTEM_PROMPT = (
    "You are a remote sensing analyst assistant. Always answer in 1-3 full "
    "sentences: give the direct answer first, then support it with visual "
    "evidence from the image (location, color, shape, count, or context). "
    "Never answer with a single word or number alone."
)


def load_pipeline(use_adapter=True):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, trust_remote_code=True, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
    )

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
        quantization_config=bnb_config, trust_remote_code=True,
    )

    if use_adapter:
        print(f"Loading LoRA adapter from: {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    else:
        print("Skipping LoRA adapter — testing BASE model only (diagnostic mode).")
        model = base_model

    model.eval()
    return model, processor


def build_fewshot_conversation(question):
    """Builds a conversation with text-only example Q&A pairs before the
    real image question, so the model imitates the answer style."""
    conversation = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]

    for ex in FEWSHOT_EXAMPLES:
        conversation.append({"role": "user", "content": [{"type": "text", "text": ex["question"]}]})
        conversation.append({"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]})

    # the real question, this time WITH the image
    conversation.append({
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    })
    return conversation


@torch.inference_mode()
def answer_vqa(model, processor, image_path, question, max_new_tokens=150):
    image = Image.open(image_path).convert("RGB")
    conversation = build_fewshot_conversation(question)

    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    confidence = 0.85 if len(answer.split()) >= 4 else 0.5
    return {"answer": answer, "confidence": confidence}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--question")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--no-adapter", action="store_true",
                         help="Skip the fine-tuned LoRA adapter — test the base model alone "
                              "to check whether fine-tuning is overriding few-shot prompting.")
    args = parser.parse_args()

    if not args.interactive and (not args.image or not args.question):
        parser.error("--image and --question are required unless --interactive is used")

    model, processor = load_pipeline(use_adapter=not args.no_adapter)

    if not args.interactive:
        print(answer_vqa(model, processor, args.image, args.question))
    else:
        print("Model ready. Type 'quit' to exit.")
        while True:
            image_path = input("\nImage path: ").strip().strip('"')
            if image_path.lower() in {"quit", "exit", "q"}:
                break
            if not Path(image_path).is_file():
                print(f"Image not found: {image_path}")
                continue
            question = input("Question: ").strip()
            if question.lower() in {"quit", "exit", "q"}:
                break
            print(answer_vqa(model, processor, image_path, question))