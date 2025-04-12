# Run the training script with the given arguments
python -m src.train \
    --data_path "ECE689AdvDL/DIPDE/kappa_field_pair-32x32N-255x255E-train:latest" \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-5 \
    --latent_dim 128 \
    --num_augmentations 3 \
    --snr_db_min 0 \
    --snr_db_max 20 \
    --reshape_nx 64 \
    --reshape_ny 64 \
    --wandb_project "DIPDE" \
    --wandb_entity "ECE689AdvDL" \
    --is_artifact \
    --device "cuda" \
    --save_path "models/kappa_denoiser_stoch.pt"
