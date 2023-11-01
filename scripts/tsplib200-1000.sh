export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u diffusion/train.py \
  --task "tsp" \
  --wandb_logger_name "tsplib200-1000" \
  --storage_path "./" \
  --validation_split "data/tsp/tsp500_concorde.txt" \
  --test_split "data/tsp/tsplib200-1000.txt" \
  --validation_examples 8 \
  --sparse_factor 100 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --two_opt_iterations 5000 \
  --resume_weight_only \
  --ckpt_path "ckpts/tsp500_categorical.ckpt" \
  --parallel_sampling 1 \
  --rewrite_ratio 0.25 \
  --norm \
  --rewrite \
  --fp16 