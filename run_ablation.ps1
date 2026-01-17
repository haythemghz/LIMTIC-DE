# Run Ablation Study
$epochs = 2
$batch_size = 32

# echo "Starting Experiment 1: All Views"
# python train_v2.py --model vit --view_mode all --epochs $epochs --batch_size $batch_size --save_dir results_ablation

echo "Starting Experiment 2: Full View Only"
python train_v2.py --model vit --view_mode full --epochs $epochs --batch_size $batch_size --save_dir results_ablation

echo "Starting Experiment 3: Side View Only"
python train_v2.py --model vit --view_mode side --epochs $epochs --batch_size $batch_size --save_dir results_ablation

echo "Ablation Study Completed."
