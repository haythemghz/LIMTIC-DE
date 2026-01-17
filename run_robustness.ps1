<<<<<<< HEAD
# Run Robustness Validation
$epochs = 2
$batch_size = 32

echo "Starting Experiment 5 (Run 1/3) Seed 42"
python train_v2.py --model vit --view_mode all --seed 42 --epochs $epochs --batch_size $batch_size --save_dir results_robustness

echo "Starting Experiment 5 (Run 2/3) Seed 100"
python train_v2.py --model vit --view_mode all --seed 100 --epochs $epochs --batch_size $batch_size --save_dir results_robustness

echo "Starting Experiment 5 (Run 3/3) Seed 2024"
python train_v2.py --model vit --view_mode all --seed 2024 --epochs $epochs --batch_size $batch_size --save_dir results_robustness

echo "Robustness Validation Completed."
=======
# Run Robustness Validation
$epochs = 2
$batch_size = 32

echo "Starting Experiment 5 (Run 1/3) Seed 42"
python train_v2.py --model vit --view_mode all --seed 42 --epochs $epochs --batch_size $batch_size --save_dir results_robustness

echo "Starting Experiment 5 (Run 2/3) Seed 100"
python train_v2.py --model vit --view_mode all --seed 100 --epochs $epochs --batch_size $batch_size --save_dir results_robustness

echo "Starting Experiment 5 (Run 3/3) Seed 2024"
python train_v2.py --model vit --view_mode all --seed 2024 --epochs $epochs --batch_size $batch_size --save_dir results_robustness

echo "Robustness Validation Completed."
>>>>>>> origin/main
