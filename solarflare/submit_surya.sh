#!/bin/bash
#SBATCH -J surya_campaign
#SBATCH -p mi3258x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o logs/surya_%j.out
#SBATCH -e logs/surya_%j.err

source $WORK/venv/bin/activate
cd $HOME/solarflare
python -u scripts/12_cache_surya_campaign.py \
  --weights-dir $WORK/surya_weights \
  --scratch $WORK/sdo_frames \
  --workers 10 --prefetch 20 \
  --time-budget-s 39600 --max-pair-gap-min 0
