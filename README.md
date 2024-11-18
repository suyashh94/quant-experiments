- Changing your base model you want to quantize --> Go to config.py and change model id. This integrates with transformers library and fetches the model from there. 
- Available types of quantization --> Go to config.py and look at keys of ```quantization_recipes```
- Quantize a model by using one of the recipes e.g. W4A16  --> python quantize.py --quantize_method W4A16
- Evaluate a model by using lm_eval.sh (make sure you have the right conda environment activated when you run this)

