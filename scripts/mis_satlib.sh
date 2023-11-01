export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

python -u diffusion/train.py \
  --task "mis" \
  --wandb_logger_name "mis_satlib" \
  --do_test_only \
  --storage_path './' \
  --training_split "data/mis/sat/test/*gpickle" \
  --validation_split "data/mis/sat/test/*gpickle" \
  --test_split "data/mis/sat/test/*gpickle" \
  --batch_size 4 \
  --hidden_dim 128 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 20 \
  --parallel_sampling 4 \
  --sequential_sampling 4 \
  --ckpt_path "ckpts/mis_er_categorical.ckpt" \
  --rewrite \
  --norm \
  --rewrite_ratio 0.1