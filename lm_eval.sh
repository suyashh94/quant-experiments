lm_eval --model vllm \
  --model_args pretrained="./Phi-3.5-mini-instruct-W8A16",add_bos_token=true \
  --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,gsm8k \
  --num_fewshot 5 \
  --limit 500 \
  --batch_size 2 \
  --output_path 'eval-results/'