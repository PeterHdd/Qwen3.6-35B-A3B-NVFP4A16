"""
Quantize Qwen/Qwen3.6-35B-A3B to NVFP4 with llm-compressor.

Two variants:
  --scheme w4a16  (default) NVFP4 weights, 16-bit activations. Best fit for RTX PRO 6000 (sm120):
                   vLLM serves NVFP4 MoE experts via the Marlin W4A16 path there anyway, so
                   weight-only gives the same speed with less quantization error.
  --scheme w4a4   NVFP4 weights AND activations. This reproduces RedHatAI's recipe, useful as a
                   sanity check (your result should land close to their numbers).

Based on the RedHatAI/Qwen3.6-35B-A3B-NVFP4 creation script.
Needs: llm-compressor from main, transformers >= 5. One 96 GB GPU + ~150 GB system RAM is enough.
Run it in the quantization environment from setup.sh, not the vLLM one: they need
different compressed-tensors versions.

Calibration: w4a16 is data-free. llm-compressor computes the FP4 scales directly from the
weights, so --samples and --seq-len have no effect and the run takes ~3 minutes.
w4a4 does use the calibration data, to measure activation ranges.

Usage:
  python quantize.py                 # w4a16 -> ./Qwen3.6-35B-A3B-NVFP4A16
  python quantize.py --scheme w4a4   # w4a4  -> ./Qwen3.6-35B-A3B-NVFP4
"""
import argparse

import torch
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

parser = argparse.ArgumentParser()
parser.add_argument("--scheme", choices=["w4a16", "w4a4"], default="w4a16")
parser.add_argument("--samples", type=int, default=256, help="calibration samples")
parser.add_argument("--seq-len", type=int, default=4096, help="max calibration sequence length")
args = parser.parse_args()

SCHEME = {"w4a16": "NVFP4A16", "w4a4": "NVFP4"}[args.scheme]
SAVE_DIR = f"Qwen3.6-35B-A3B-{SCHEME}"

# Layers that stay in BF16. These are small but sensitive: output head, vision tower,
# MoE router gates, embeddings, and the linear-attention (GatedDeltaNet) blocks.
IGNORE = [
    "re:.*lm_head",
    "re:visual.*",
    "re:model.visual.*",
    "re:.*mlp.gate$",
    "re:.*embed_tokens$",
    "re:.*shared_expert_gate$",
    "re:.*linear_attn.*",
]

print(f"Loading {MODEL_ID} in BF16 ...")
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(targets="Linear", scheme=SCHEME, ignore=IGNORE)

# Calibration data: real chat conversations, formatted with the model's own chat template.
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{args.samples}]")
ds = ds.select_columns(["messages"]).shuffle(seed=42)


def preprocess(example):
    messages = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in example["messages"]
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        processor_kwargs={
            "return_tensors": "pt",
            "padding": False,
            "truncation": True,
            "max_length": args.seq_len,
            "add_special_tokens": False,
        },
    )


ds = ds.map(preprocess, batched=False, remove_columns=ds.column_names)


def collate(batch):
    assert len(batch) == 1
    return {k: torch.tensor(v) for k, v in batch[0].items()}


print(f"Quantizing with scheme={SCHEME}, {args.samples} samples x {args.seq_len} tokens ...")
oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=args.seq_len,
    num_calibration_samples=args.samples,
    # Route calibration tokens through ALL 256 experts so rarely-used experts
    # still get proper scale statistics (important for MoE quality).
    moe_calibrate_all_experts=True,
    data_collator=collate,
)

model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

# The MTP (multi-token prediction) draft layers aren't loaded by this model class,
# so copy them unchanged from the original checkpoint. vLLM uses them for speculative decoding.
save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=SAVE_DIR)

print(f"Done -> ./{SAVE_DIR}")
