# Unified Experiment Runner for LIMTIC-DE
# Implements the 5-independent run protocol over 5 seeds

$EPOCHS = 2
$BATCH_SIZE = 32
$SEEDS = @(42, 100, 2024, 1234, 5678)

Write-Host "=== LIMTIC-DE 5-Seed Independent Run Protocol ===" -ForegroundColor Cyan

foreach ($seed in $SEEDS) {
    Write-Host "`n--- Running Seed $seed ---" -ForegroundColor Yellow
    
    # 1. ViT Baseline
    Write-Host "Training ViT..."
    python train_v2.py --model vit --view_mode all --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $seed --save_dir results_robustness/vit_seed_$seed
    
    # 2. ResNet50 (+ Augmentation)
    Write-Host "Training ResNet50..."
    python train_v2.py --model resnet --view_mode all --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $seed --augment --save_dir results_robustness/resnet_seed_$seed
    
    # 3. EfficientNet-B0
    Write-Host "Training EfficientNet-B0..."
    python train_v2.py --model efficientnet --view_mode all --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $seed --save_dir results_robustness/effnet_seed_$seed
}

Write-Host "`nAll 5 seeds completed for all baseline architectures." -ForegroundColor Green
Write-Host "You can now run 'python calculate_significance.py' to generate statistical reports." -ForegroundColor Green
