<<<<<<< HEAD
# Run Augmentation Study
$epochs = 2
$batch_size = 32

echo "Starting Experiment 4.1: ResNet50 with Augmentation"
python train_v2.py --model resnet --view_mode all --augment --epochs $epochs --batch_size $batch_size --save_dir results_augmentation

echo "Starting Experiment 4.2: ViT with Augmentation"
python train_v2.py --model vit --view_mode all --augment --epochs $epochs --batch_size $batch_size --save_dir results_augmentation

echo "Augmentation Study Completed."
=======
# Run Augmentation Study
$epochs = 2
$batch_size = 32

echo "Starting Experiment 4.1: ResNet50 with Augmentation"
python train_v2.py --model resnet --view_mode all --augment --epochs $epochs --batch_size $batch_size --save_dir results_augmentation

echo "Starting Experiment 4.2: ViT with Augmentation"
python train_v2.py --model vit --view_mode all --augment --epochs $epochs --batch_size $batch_size --save_dir results_augmentation

echo "Augmentation Study Completed."
>>>>>>> origin/main
