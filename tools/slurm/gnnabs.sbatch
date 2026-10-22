#!/bin/bash
#SBATCH --job-name=gnnabs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=analysis/kcdc/gnnabs_qgs4_slurm-%j.out
# Graph network over KASCADE stations with ABSOLUTE station positions (no core-relative input),
# 40 epochs with early stopping: the ablation of the core geometry. QGSJET-II-04, lgE in [5,6].
cd "${SLURM_SUBMIT_DIR:-.}"     # submit from the repository root: sbatch tools/slurm/gnnabs.sbatch
ENCODER=gnnabs EPOCHS=40 ${PY:-python3} -u analysis/kcdc/run_kcdc_cnn.py
