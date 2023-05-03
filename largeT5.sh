#export ALL_PROXY=socks5://10.1.17.172:7890; export http_proxy=http://10.1.17.172:7890; export https_proxy=http://10.1.17.172:7890
len=${#1}
len=$((len+1))
nproc_per_node=$((len/2))
gas=$((16/nproc_per_node))
CUDA_VISIBLE_DEVICES=$1 \
python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port=$5 finetune_trainer.py \
  --data_dir /home/disk2/xcruan/doc2dial/UniGDD/task_rwth/$3 \
  --cache_dir ./cache \
  --output_dir ./output/$4/ \
  --num_train_epochs $2 \
  --model_name_or_path t5-large \
  --learning_rate 1e-4 \
  --adam_epsilon 1e-06 \
  --do_train \
  --do_eval false \
  --temp_start 1.0 \
  --temp_end 1.0 \
  --scheduler linear \
  --eval_beams 2 \
  --per_device_train_batch_size=1 \
  --per_device_eval_batch_size=12 \
  --gradient_accumulation_steps=$gas \
  --max_source_length 1024 \
  --max_target_length 75 \
  --task translation \
  --warmup_steps 500 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_steps 200 \
  --predict_with_generate \
  --save_total_limit 3 \
  --generation_max_length 75 \
  --generation_num_beams 2 \
  --overwrite_output_dir \
  --seed 42

#  --seed 21

  # --add_tokens \
