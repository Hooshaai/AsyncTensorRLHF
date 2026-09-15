#!/usr/bin/env bash
# Launch the orchestrator process using Ray (or as a standalone async task)

# Example usage:
#   scripts/launch_orchestrator.sh <model_path>

MODEL_PATH=$1

if [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 <model_path>"
  exit 1
fi

# Ensure Ray is running
if ! ray status > /dev/null 2>&1; then
  ray start --head
fi

# Run orchestrator as a background Ray task
ray exec "import src.orchestrator.scheduler as sch; from src.rollout.version_manager import VersionManager; vm = VersionManager(); async def sync(v): pass; orch = sch.Orchestrator(vm, sync); asyncio.run(orch.run())" --address=auto &

echo "Orchestrator launched for model at $MODEL_PATH"
