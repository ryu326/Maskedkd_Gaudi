python train.py --name imagenet1k_TF --dataset imagenet1K \
--data_path /workspace/imagenet --model_type ViT-B_16 \
--num_steps 20000 --eval_every 1000 --train_batch_size 32 \
--gradient_accumulation_steps 1 --img_size 384 --learning_rate 0.06