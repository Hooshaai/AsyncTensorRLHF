#!/usr/bin/env bash
# Launch rollout workers using Ray

# Example usage:
#   scripts/launch_rollout.sh <num_workers> <model_path>

NUM_WORKERS=$1
MODEL_PATH=$2

if [ -z "$NUM_WORKERS" ] || [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 <num_workers> <model_path>"
  exit 1
fi

# Start Ray head (if not already running)
if ! ray status > /dev/null 2>&1; then
  ray start --head
fi

# Launch rollout actors
for i in $(seq 1 $NUM_WORKERS); do
  ray exec "import src.rollout.async_engine as ae; from src.buffer.replay_buffer import BoundedReplayBuffer; buffer = BoundedReplayBuffer(); engine = ae.AsyncEngine(engine=None, buffer=buffer);" --address=auto &
  # The above is a placeholder; replace with actual Ray actor creation script.
done

echo "Launched $NUM_WORKERS rollout workers for model at $MODEL_PATH"
