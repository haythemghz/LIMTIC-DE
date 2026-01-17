# Run EfficientNet-B0 baseline (2 epochs) for comparative benchmarking

$EPOCHS = 2
$BATCH_SIZE = 32

Write-Host "Running EfficientNet-B0 Baseline" -ForegroundColor Cyan

# EfficientNet-B0 (Seed 42)
python train_v2.py --model efficientnet --view_mode all --epochs $EPOCHS --batch_size $BATCH_SIZE --seed 42 --save_dir results_ablation

Write-Host "EfficientNet-B0 Completed." -ForegroundColor Green
