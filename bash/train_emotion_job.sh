#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_emotion
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 12:00
#BSUB -o outputs/emotion_train_%J.out
#BSUB -e outputs/emotion_train_%J.err

# Create outputs directory
mkdir -p outputs
mkdir -p models

# Load Python module
module load python3/3.12.11

# Add UV to PATH
export PATH="$HOME/.local/bin:$PATH"

# Navigate to project directory
cd ~/projects/militol_audio_analysis

# Activate UV environment
source .venv/bin/activate

# Run training script
python scripts/train_emotion.py