export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u diffusion/train.py \
  --task "tsp" \
  --wandb_logger_name "tsp50" \
  --storage_path "./" \
  --validation_split "data/tsp/tsp50_concorde.txt" \
  --test_split "data/tsp/tsp50_concorde.txt" \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --two_opt_iterations 1000 \
  --resume_weight_only \
  --ckpt_path "ckpts/tsp50_categorical.ckpt" \
  --parallel_sampling 1 \
  --rewrite_ratio 0.4 \
  --norm \
  --rewrite