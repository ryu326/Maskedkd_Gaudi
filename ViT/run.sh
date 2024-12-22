# imagenet21k pre-train
# wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz

# # imagenet21k pre-train + imagenet2012 fine-tuning
# wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz

# python train.py --name imagenet1k_TF --dataset imagenet1K \
# --data_path /workspace/imagenet --model_type ViT-B_16 \
# --num_steps 20000 --eval_every 1000 --train_batch_size 32 \
# --gradient_accumulation_steps 1 --img_size 384 --learning_rate 0.06

nohup mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train.py --name ViT-L_16_no_distill --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-L_16 --num_steps 20000 --eval_every 1000 --train_batch_size 64 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
--log_path logs/ViT-L_16_no_distill > running_ViT-L_16_no_distill.txt 2>&1 &

wait $!

nohup mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train_distill.py --name maskedkd_98_myT --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-B_16 --num_steps 20000 --eval_every 1000 --train_batch_size 64 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
--teacher_model_type ViT-L_16 --teacher_pretrained_dir output/ViT-L_16_no_distill_checkpoint.pth \
--maskedkd --len_num_keep 98 \
--log_path logs/log_maskedkd_98_myT > running_log_maskedkd_98_myT.txt 2>&1 &

wait $!

nohup mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train_distill.py --name naive_distill_myT --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-B_16 --num_steps 20000 --eval_every 1000 --train_batch_size 64 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
--teacher_model_type ViT-L_16 --teacher_pretrained_dir output/ViT-L_16_no_distill_checkpoint.pth \
--log_path logs/log_naive_myT > running_log_naive_my_T.txt 2>&1 &

wait $!

nohup mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train_distill.py --name maskedkd_49 --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-B_16 --num_steps 20000 --eval_every 1000 --train_batch_size 64 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
--teacher_model_type ViT-L_16 --teacher_pretrained_dir pretrained_models/ViT-L_16-224.npz \
--maskedkd --len_num_keep 49 \
--log_path logs/log_maskedkd_49 > running_log_maskedkd_49.txt 2>&1 &

wait $!

nohup mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train_distill.py --name maskedkd_147 --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-B_16 --num_steps 20000 --eval_every 1000 --train_batch_size 64 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
--teacher_model_type ViT-L_16 --teacher_pretrained_dir pretrained_models/ViT-L_16-224.npz \
--maskedkd --len_num_keep 147 \
--log_path logs/log_maskedkd_147 > running_log_maskedkd_147.txt 2>&1 &

wait $!

nohup mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train_distill.py --name naive_distill --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-B_16 --num_steps 20000 --eval_every 1000 --train_batch_size 64 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
--teacher_model_type ViT-L_16 --teacher_pretrained_dir pretrained_models/ViT-L_16-224.npz \
--log_path logs/log_naive > running_log_naive.txt 2>&1 &

wait $!

mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u validate.py --name validate_ViT-L_16 --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-L_16 --num_steps 20000 --eval_every 1000 --train_batch_size 64 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
--pretrained_dir pretrained_models/ViT-L_16-224.npz \
--log_path logs/validate_ViT-L_16

mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train.py --name validate_ViT-L_16 --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-L_16 --num_steps 3 --eval_every 1 --train_batch_size 64 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
--pretrained_dir pretrained_models/ViT-L_16-224.npz \
--log_path logs/validate_ViT-L_16



wait $!