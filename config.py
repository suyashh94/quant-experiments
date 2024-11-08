from llmcompressor.modifiers.quantization import QuantizationModifier,GPTQModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


quantization_recipes = {
        "W8A16" : {
            "recipe" : QuantizationModifier(targets="Linear", scheme="W8A16", ignore=["lm_head"]),
            "needs_calibration" : False
        }, 
        "W8A8" : {
            "recipe" : QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
            "needs_calibration" : False
        },
        "W4A16" : {
            "recipe" : QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
            "needs_calibration" : False
        },
        "W4A8" : {
            "recipe" : QuantizationModifier(targets="Linear", scheme="W4A8", ignore=["lm_head"]),
            "needs_calibration" : False
        },
        "FP8_DYNAMIC" : {
            "recipe" : QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]),
            "needs_calibration" : False
        },
        "FP8" : {
            "recipe" : QuantizationModifier(targets="Linear", scheme="FP8", ignore=["lm_head"]),
            "needs_calibration" : True
        }, 
        "GPTQ-W4A16"  :{
            "recipe" : GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
            "needs_calibration" : True
        }, 
        "GPTQ-W8A8" : {
            "recipe" : GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
            "needs_calibration" : True
        },     
    }
