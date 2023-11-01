export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=2

python -u diffusion/train.py \
  --task "tsp" \
  --wandb_logger_name "tsplib50-200" \
  --storage_path "./" \
  --validation_split "data/tsp/tsp100_concorde.txt" \
  --test_split "data/tsp/tsplib50-200.txt" \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --two_opt_iterations 1000 \
  --resume_weight_only \
  --ckpt_path "ckpts/tsp100_categorical.ckpt" \
  --parallel_sampling 1 \
  --rewrite_ratio 0.25 \
  --norm \
  --rewrite 