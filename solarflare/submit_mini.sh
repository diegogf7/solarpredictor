#!/bin/bash
#SBATCH -J mini
#SBATCH -p mi3258x
#SBATCH -N 1 -n 1 -t 01:00:00
#SBATCH -o logs/mini_%j.out
#SBATCH -e logs/mini_%j.err
source $WORK/venv/bin/activate
cd $HOME/solarflare
python -u scripts/16_mini_pipeline.py   --surya artifacts/caches/surya_campaign_val.pt   --readouts $WORK/readouts/readouts_val
