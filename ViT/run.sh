# imagenet21k pre-train
# wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz

# # imagenet21k pre-train + imagenet2012 fine-tuning
# wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz

# python train.py --name imagenet1k_TF --dataset imagenet1K \
# --data_path /workspace/imagenet --model_type ViT-B_16 \
# --num_steps 20000 --eval_every 1000 --train_batch_size 32 \
# --gradient_accumulation_steps 1 --img_size 384 --learning_rate 0.06

nohup mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train.py --name imagenet1k_TF --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-B_16 --num_steps 20000 --eval_every 1000 --train_batch_size 64 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
--log_path log_no_distill > running_log_no_distill.txt 2>&1 && \

nohup mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train_distill.py --name imagenet1k_TF --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-B_16 --num_steps 20000 --eval_every 1000 --train_batch_size 64 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
--teacher_model_type ViT-L_16 --teacher_pretrained_dir pretrained_models/ViT-L_16-224.npz \
--log_path log_naive_distill > running_log_naive_distill.txt 2>&1 && \

nohup mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
python -u train_distill.py --name imagenet1k_TF --dataset imagenet1K --data_path /workspace/imagenet \
--model_type ViT-B_16 --num_steps 20000 --eval_every 1000 --train_batch_size 64 \
--gradient_accumulation_steps 2 --img_size 224 --learning_rate 0.06 --autocast \
--teacher_model_type ViT-L_16 --teacher_pretrained_dir pretrained_models/ViT-L_16-224.npz \
--maskedkd --len_num_keep 98 \
--log_path log_maskedkd_distill > running_log_maskedkd_distill.txt 2>&1


