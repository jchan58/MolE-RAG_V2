#!/bin/sh
CUDA_VISIBLE_DEVICES=5
WANDB_API_KEY=$WANDB_API_KEY python main_transformer_smiles2caption.py --mode train --model_size large --MoMuK --use_3d