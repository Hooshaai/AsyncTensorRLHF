#!/usr/bin/env bash
# Launch trainer workers using Ray

# Example usage:
#   scripts/launch_trainer.sh <num_workers> <model_path>

NUM_WORKERS=$1
MODEL_PATH=$2

if [ -z "$NUM_WORKERS" ] || [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 <num_workers> <model_path>"
  exit 1
fi

# Ensure Ray is running
if ! ray status > /dev/null 2>&1; then
  ray start --head
fi

for i in $(seq 1 $NUM_WORKERS); do
  ray exec "import src.trainer.trainer_worker as tw; tw.start_trainer('$MODEL_PATH')" --address=auto &
  # Placeholder: replace with actual Ray actor creation.
done

echo "Launched $NUM_WORKERS trainer workers for model at $MODEL_PATH"
