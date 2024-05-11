export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u diffusion/train.py \
  --task "tsp" \
  --wandb_logger_name "tsp1000" \
  --storage_path "./" \
  --validation_split "data/tsp/tsp1000_concorde.txt" \
  --test_split "data/tsp/tsp1000_concorde.txt" \
  --validation_examples 8 \
  --sparse_factor 100 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 20 \
  --resume_weight_only \
  --ckpt_path "ckpts/tsp1000_categorical.ckpt" \
  --two_opt_iterations 5000 \
  --parallel_sampling 1 \
  --rewrite_ratio 0.4 \
  --norm \
  --rewrite \
  --fp16