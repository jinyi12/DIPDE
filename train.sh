# Run the training script with the given arguments
python -m src.train \
    --data_path "ECE689AdvDL/DIPDE/kappa_field_pair-32x32-to-256x256:latest" \
    --batch_size 32 \
    --epochs 20 \
    --lr 1e-5 \
    --latent_dim 128 \
    --num_augmentations 2 \
    --snr_db_min 5 \
    --snr_db_max 20 \
    --reshape_nx 64 \
    --reshape_ny 64 \
    --wandb_project "DIPDE" \
    --wandb_entity "ECE689AdvDL" \
    --is_artifact \
    --device "cuda" \
    --save_path "kappa_denoiser_stoch.pt"
