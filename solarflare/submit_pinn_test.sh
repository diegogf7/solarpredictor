#!/bin/bash
#SBATCH -J pinn_test
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o logs/pinn_test_%j.out
#SBATCH -e logs/pinn_test_%j.err
source $WORK/venv/bin/activate
cd $HOME/solarflare

python -u scripts/13_fetch_campaign_cubes.py \
  --positions flare_forecaster/cache/campaign_positions_test.json \
  --out flare_forecaster/cache/cubes_test

python -u scripts/14_pinn_campaign.py \
  --cubes flare_forecaster/cache/cubes_test \
  --out artifacts/caches/readouts_test \
  --device cuda --time-budget-s 39600
