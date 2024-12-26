# imagenet21k pre-train
# wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz

# # imagenet21k pre-train + imagenet2012 fine-tuning
# wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz

# python train.py --name imagenet1k_TF --dataset imagenet1K \
# --data_path /workspace/imagenet --model_type ViT-B_16 \
# --num_steps 20000 --eval_every 1000 --train_batch_size 32 \
# --gradient_accumulation_steps 1 --img_size 384 --learning_rate 0.06

# nohup mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
# python -u train.py --name ViT-L_16 --dataset imagenet1K --data_path /workspace/imagenet \
# --model_type ViT-L_16 --num_steps 20000 --eval_every 1000 --train_batch_size 64 \
# --gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
# --pretrained_dir pretrained_models/ViT-L_16-224.npz \
# --log_path logs/ViT-L_16 > running_logs/ViT-L_16.txt 2>&1 &

# wait $!

# nohup mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
# python -u train_distill.py --name L2B_naive_distill_ft --dataset imagenet1K --data_path /workspace/imagenet \
# --model_type ViT-B_16 --num_steps 20000 --eval_every 1000 --train_batch_size 64 \
# --pretrained_dir pretrained_models/ViT-B_16-224.npz \
# --gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
# --teacher_model_type ViT-L_16 --teacher_pretrained_dir output/ViT-L_16_checkpoint.pth \
# --log_path logs/L2B_naive_distill_ft > running_logs/L2B_naive_distill_ft.txt 2>&1 &

# wait $!

# nohup mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
# python -u train_distill.py --name L2B_maskedkd_98_ft --dataset imagenet1K --data_path /workspace/imagenet \
# --model_type ViT-B_16 --num_steps 20000 --eval_every 1000 --train_batch_size 64 \
# --pretrained_dir pretrained_models/ViT-B_16-224.npz \
# --gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
# --teacher_model_type ViT-L_16 --teacher_pretrained_dir output/ViT-L_16_checkpoint.pth \
# --maskedkd --len_num_keep 98 \
# --log_path logs/L2B_maskedkd_98_ft > running_logs/L2B_maskedkd_98_ft.txt 2>&1 &

# wait $!

# nohup mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
# python -u train_distill.py --name L2B_maskedkd_147_ft --dataset imagenet1K --data_path /workspace/imagenet \
# --model_type ViT-B_16 --num_steps 20000 --eval_every 1000 --train_batch_size 64 \
# --pretrained_dir pretrained_models/ViT-B_16-224.npz \
# --gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
# --teacher_model_type ViT-L_16 --teacher_pretrained_dir output/ViT-L_16_checkpoint.pth \
# --maskedkd --len_num_keep 147 \
# --log_path logs/L2B_maskedkd_147_ft > running_logs/L2B_maskedkd_147_ft.txt 2>&1 &

# wait $!

# nohup mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
# python -u train_distill.py --name L2B_maskedkd_49_ft --dataset imagenet1K --data_path /workspace/imagenet \
# --model_type ViT-B_16 --num_steps 20000 --eval_every 1000 --train_batch_size 64 \
# --pretrained_dir pretrained_models/ViT-B_16-224.npz \
# --gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
# --teacher_model_type ViT-L_16 --teacher_pretrained_dir output/ViT-L_16_checkpoint.pth \
# --maskedkd --len_num_keep 147 \
# --log_path logs/L2B_maskedkd_49_ft > running_logs/L2B_maskedkd_49_ft.txt 2>&1 &

# wait $!

mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train.py --name S_16_scratch_2 --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-S_16 --num_steps 100000 --eval_every 1000 --train_batch_size 1024 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 5e-4 --autocast \
--log_path logs/VIT-S_16_scratch_2 | tee running_logs/S_16_scratch_2.txt

mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train_distill.py --name B2S_maskedkd_98_scratch_2 --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-S_16 --num_steps 100000 --eval_every 1000 --train_batch_size 1024 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 5e-4 --autocast \
--teacher_model_type ViT-B_16 --teacher_pretrained_dir output/L2B_maskedkd_98_ft_checkpoint.pth \
--maskedkd --len_num_keep 98 \
--log_path logs/B2S_maskedkd_98_scratch_2 | tee running_logs/B2S_maskedkd_98_scratch_2.txt

mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train_distill.py --name B2S_naive_distill_scratch_2 --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-S_16 --num_steps 100000 --eval_every 1000 --train_batch_size 1024 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 5e-4 --autocast \
--teacher_model_type ViT-B_16 --teacher_pretrained_dir output/L2B_maskedkd_98_ft_checkpoint.pth \
--log_path logs/B2S_naive_distill_scratch_2 | tee running_logs/B2S_naive_distill_scratch_2.txt

mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train.py --name S_16_scratch --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-S_16 --num_steps 10000000 --eval_every 1000 --train_batch_size 64 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 5e-4 --autocast \
--log_path logs/VIT-S_16_scratch | tee running_logs/S_16_scratch.txt

mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train_distill.py --name B2S_maskedkd_147_scratch_2 --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-S_16 --num_steps 100000 --eval_every 1000 --train_batch_size 1024 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 5e-4 --autocast \
--teacher_model_type ViT-B_16 --teacher_pretrained_dir output/L2B_maskedkd_98_ft_checkpoint.pth \
--maskedkd --len_num_keep 147 \
--log_path logs/B2S_maskedkd_147_scratch_2 | tee running_logs/B2S_maskedkd_147_scratch_2.txt

mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train_distill.py --name L2S_maskedkd_98_scratch_2 --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-S_16 --num_steps 100000 --eval_every 1000 --train_batch_size 1024 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 5e-4 --autocast \
--teacher_model_type ViT-L_16 --teacher_pretrained_dir output/ViT-L_16_checkpoint.pth \
--maskedkd --len_num_keep 98 \
--log_path logs/L2S_maskedkd_98_scratch_2 | tee running_logs/L2S_maskedkd_98_scratch_2.txt

mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train_distill.py --name L2S_naive_distill_scratch_2 --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-S_16 --num_steps 100000 --eval_every 1000 --train_batch_size 1024 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 5e-4 --autocast \
--teacher_model_type ViT-L_16 --teacher_pretrained_dir output/ViT-L_16_checkpoint.pth \
--log_path logs/L2S_naive_distill_scratch_2 | tee running_logs/L2S_naive_distill_scratch_2.txt

mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train_distill.py --name L2S_maskedkd_147_scratch_2 --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-S_16 --num_steps 100000 --eval_every 1000 --train_batch_size 1024 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 5e-4 --autocast \
--teacher_model_type ViT-L_16 --teacher_pretrained_dir output/ViT-L_16_checkpoint.pth \
--maskedkd --len_num_keep 147 \
--log_path logs/L2S_maskedkd_147_scratch_2 | tee running_logs/L2S_maskedkd_147_scratch_2.txt


# mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
# python -u train.py --name test --dataset imagenet1K --data_path /workspace/imagenet \
# --model_type ViT-L_16 --num_steps 1 --eval_every 1 --train_batch_size 64 \
# --gradient_accumulation_steps 2 --img_size 224 --learning_rate 5e-4 --autocast \
# --resume output/ViT-L_16_checkpoint.pth \
# --opt sgd \
# --log_path logs/test



# nohup mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
# python -u train_distill.py --name L2B_maskedkd_49_scratch --dataset imagenet1K --data_path /workspace/imagenet \
# --model_type ViT-B_16 --num_steps 120000 --eval_every 1000 --train_batch_size 64 \
# --gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
# --teacher_model_type ViT-L_16 --teacher_pretrained_dir output/ViT-L_16_checkpoint.pth \
# --maskedkd --len_num_keep 147 \
# --log_path logs/L2B_maskedkd_49_scratch > running_logs/L2B_maskedkd_49_scratch.txt 2>&1 &

# wait $!