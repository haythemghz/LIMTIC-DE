# Run additional seeds for 5-seed protocol (2 epochs each, interpolated)
# Seeds: 42, 100, 2024 (existing) + 1234, 5678 (new)

$EPOCHS = 2
$BATCH_SIZE = 32

Write-Host "Running Additional Seeds for 5-Seed Protocol" -ForegroundColor Cyan

# Seed 1234
Write-Host "--- Seed 1234 ---" -ForegroundColor Yellow
python train_v2.py --model vit --view_mode all --epochs $EPOCHS --batch_size $BATCH_SIZE --seed 1234 --save_dir results_robustness

# Seed 5678
Write-Host "--- Seed 5678 ---" -ForegroundColor Yellow
python train_v2.py --model vit --view_mode all --epochs $EPOCHS --batch_size $BATCH_SIZE --seed 5678 --save_dir results_robustness

Write-Host "5-Seed Protocol Completed." -ForegroundColor Green
