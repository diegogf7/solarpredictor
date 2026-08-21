#!/bin/bash
#SBATCH -J pinn_train
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o logs/pinn_train_%j.out
#SBATCH -e logs/pinn_train_%j.err
source $WORK/venv/bin/activate
cd $HOME/solarflare

python -u scripts/13_fetch_campaign_cubes.py \
  --positions flare_forecaster/cache/campaign_positions.json \
  --out flare_forecaster/cache/cubes

python -u scripts/14_pinn_campaign.py \
  --cubes flare_forecaster/cache/cubes \
  --out artifacts/caches/readouts_train \
  --device cuda --time-budget-s 39600
