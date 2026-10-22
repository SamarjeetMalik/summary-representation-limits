#!/bin/bash
#SBATCH --job-name=gnnabs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/samarjeet_malik/grapes3_tmlr/analysis/kcdc/gnnabs_qgs4_slurm-%j.out
# Graph network over KASCADE stations with ABSOLUTE station positions (no core-relative input),
# 40 epochs with early stopping: the ablation of the core geometry. QGSJET-II-04, lgE in [5,6].
cd /scratch/samarjeet_malik/grapes3_tmlr
ENCODER=gnnabs EPOCHS=40 ~/.local/share/mamba/envs/ml/bin/python -u analysis/kcdc/run_kcdc_cnn.py
