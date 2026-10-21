#!/usr/bin/env bash
# Benchmark one model: size, quality (MMLU-Pro + GSM8K via lm-eval), speed (vLLM bench).
# Run it for YOUR build and for each competitor ON THE SAME MACHINE, so the numbers are comparable.
#
# Usage:
#   ./benchmark.sh ./Qwen3.6-35B-A3B-NVFP4A16      # your build (local path)
#   ./benchmark.sh kyaky/Qwen3.6-35B-A3B-NVFP4     # competitors (HF ids)
#   ./benchmark.sh nvidia/Qwen3.6-35B-A3B-NVFP4
#   ./benchmark.sh RedHatAI/Qwen3.6-35B-A3B-NVFP4
#   ./benchmark.sh Qwen/Qwen3.6-35B-A3B            # BF16 original = 100% reference
#
# Results land in ./results/<model-name>/
#
# Optional settings (environment variables):
#   SKIP_SPEED=1          skip the speed test (e.g. it already ran)
#   SKIP_MMLU=1           skip MMLU-Pro
#   SKIP_GSM8K=1          skip GSM8K
#   MMLU_TASK=<name>      lm-eval task name for MMLU-Pro   (default: mmlu_pro_llama)
#   GSM8K_TASK=<name>     lm-eval task name for GSM8K      (default: gsm8k_cot_llama)
#   Find valid names with:  lm_eval ls tasks 2>/dev/null | grep -E "^\|(mmlu_pro|gsm8k)" | cut -d'|' -f2
#
# Why the *_llama tasks: they are 0-shot chat variants that tell the model the exact
# answer format. Plain `mmlu_pro` relies on 5 few-shot examples to teach the format;
# at 0-shot the answer extractor misses most answers and the score collapses (~0.17).
#
# Example (re-run only MMLU-Pro):
#   SKIP_SPEED=1 SKIP_GSM8K=1 ./benchmark.sh ./Qwen3.6-35B-A3B-NVFP4A16
set -euo pipefail

SKIP_SPEED="${SKIP_SPEED:-0}"
SKIP_MMLU="${SKIP_MMLU:-0}"
SKIP_GSM8K="${SKIP_GSM8K:-0}"
MMLU_TASK="${MMLU_TASK:-mmlu_pro_llama}"
GSM8K_TASK="${GSM8K_TASK:-gsm8k_cot_llama}"

MODEL="$1"
if [ -d "$MODEL" ]; then
  NAME="$(basename "$MODEL")-local"
else
  NAME="$(basename "$MODEL")-$(echo "$MODEL" | cut -d/ -f1)"
fi
OUT="results/$NAME"
PORT=8000
URL="http://localhost:$PORT"
mkdir -p "$OUT"

# ---- 1. Size on disk -------------------------------------------------------
if [ -d "$MODEL" ]; then
  DIR="$MODEL"
else
  DIR="$(hf download "$MODEL" --quiet)"
fi
# -L follows the symlinks in the Hugging Face cache, so we measure the real files.
du -shL "$DIR" | tee "$OUT/size.txt"

# ---- 2. Start vLLM server ---------------------------------------------------
# Wait until the GPU is free. A previous server can take a while to release memory,
# and a new one crashes at startup if it can't claim 90% of the GPU.
for i in $(seq 1 60); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$USED" -lt 2000 ] && break
  echo "GPU still has ${USED} MiB in use, waiting for it to free up ..."
  sleep 5
done
if [ "$USED" -ge 2000 ]; then
  echo "GPU memory still in use after 5 minutes. Check nvidia-smi and stop other processes (pkill -f 'vllm serve')."
  exit 1
fi
# Thinking on (default for Qwen3.6). Text-only to skip loading the vision tower.
# --max-num-seqs 256: Qwen3.6's linear-attention layers need one state ("Mamba cache") block
# per running request. vLLM's default of 1024 doesn't fit next to the 67 GiB BF16 weights.
# The benchmarks never send more than 64 requests at once, so 256 changes no results.
vllm serve "$MODEL" \
  --served-model-name bench \
  --port $PORT \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --reasoning-parser qwen3 \
  --language-model-only \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
# On exit, stop the server and wait for it (and its engine subprocess) to fully shut down,
# so the next benchmark starts with a free GPU.
cleanup() {
  kill $SERVER_PID 2>/dev/null || true
  wait $SERVER_PID 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server (first start takes a few minutes) ..."
echo "Watch progress with: tail -f $OUT/server.log"
until curl -s "$URL/health" > /dev/null; do
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "vLLM server exited during startup. Likely cause (errors from $OUT/server.log):"
    grep -E "Error|error" "$OUT/server.log" | grep -v 'File "' | grep -v "^\s*\^" | head -10
    echo "Full log: $OUT/server.log"
    exit 1
  fi
  sleep 5
done
echo "Server ready."

# ---- 3. Speed ---------------------------------------------------------------
# Single-stream (one user) and 16 concurrent users, matching kyaky's two speed columns.
if [ "$SKIP_SPEED" = "1" ]; then
  echo "Skipping speed test (SKIP_SPEED=1)."
else
  for C in 1 16; do
    vllm bench serve \
      --backend openai-chat --endpoint /v1/chat/completions \
      --base-url "$URL" --model bench --tokenizer "$MODEL" \
      --dataset-name random --random-input-len 1024 --random-output-len 512 \
      --num-prompts $((C * 8)) --max-concurrency $C \
      --ignore-eos \
      --save-result --result-dir "$OUT" --result-filename "speed_c$C.json" \
      | tee "$OUT/speed_c$C.txt"
  done
fi

# ---- 4. Quality -------------------------------------------------------------
# Same sampling settings as RedHatAI's published eval.
GEN="do_sample=true,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=32000,presence_penalty=1.5,seed=42"
ARGS="model=bench,base_url=$URL/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=3600"

# --log_samples saves every prompt + model answer + extracted answer as JSONL,
# so you can check the answer extraction actually worked.

# MMLU-Pro: 100 questions per subject (~1.4k total), same subset size kyaky used.
if [ "$SKIP_MMLU" = "1" ]; then
  echo "Skipping MMLU-Pro (SKIP_MMLU=1)."
else
  lm_eval --model local-chat-completions --tasks "$MMLU_TASK" \
    --model_args "$ARGS" --num_fewshot 0 --apply_chat_template \
    --limit 100 --gen_kwargs "$GEN" --seed 42 --log_samples \
    --output_path "$OUT/mmlu_pro" | tee "$OUT/mmlu_pro.txt"
fi

# GSM8K: full test set (1,319 problems), report flexible-extract.
if [ "$SKIP_GSM8K" = "1" ]; then
  echo "Skipping GSM8K (SKIP_GSM8K=1)."
else
  lm_eval --model local-chat-completions --tasks "$GSM8K_TASK" \
    --model_args "$ARGS" --num_fewshot 0 --apply_chat_template \
    --gen_kwargs "$GEN" --seed 42 --log_samples \
    --output_path "$OUT/gsm8k" | tee "$OUT/gsm8k.txt"
fi

echo "Done. Results in $OUT/"