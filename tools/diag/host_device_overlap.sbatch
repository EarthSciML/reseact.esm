#!/bin/bash
#SBATCH --job-name=devovl
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
echo "=== node $(hostname) cpus=$SLURM_CPUS_PER_TASK ==="
for nd in 1 2 4 8 16; do
  echo "--- xla_force_host_platform_device_count=$nd"
  XLA_FLAGS="--xla_force_host_platform_device_count=$nd" NDEV=$nd \
    julia -t 16 --project=run-model-jl tools/diag/host_device_overlap.jl run-model-jl 2>&1 | grep -E "devices visible|concurrent"
done
