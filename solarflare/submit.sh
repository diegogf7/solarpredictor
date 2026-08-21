#!/bin/bash
#SBATCH --job-name=solarflare-cnn
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu          # check yours: run `sinfo` on the cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

mkdir -p logs

module purge
module load cuda
module load python

source ~/solarflare/venv/bin/activate

cd ~/solarflare

python train.py \
    --train_csv  solarflaredata/Train_Data_by_AR_png_224.csv \
    --val_csv    solarflaredata/Validation_Data_by_AR_png_224.csv \
    --images_dir solarflaredata/Lat60_Lon60_Nans0_png_224/ \
    --epochs 10 \
    --batch_size 32 \
    --num_workers 4
