#!/usr/bin/env bash
# Wait for harvest to finish then sync to HF bucket
while ps -p $1 > /dev/null 2>&1; do
    sleep 30
done
echo "[upload] harvest done, syncing to HF bucket..."
cd "$(dirname "$0")"
hf sync ./outputs hf://buckets/twanghcmut/evidence-ai/papers/v1
echo "[upload] done"
