from datasets import load_dataset
from transformers import AutoTokenizer

from llmcompressor.modifiers.quantization import QuantizationModifier,GPTQModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot

from config import MODEL_ID, DATASET_ID, DATASET_SPLIT, NUM_CALIBRATION_SAMPLES,\
    MAX_SEQUENCE_LENGTH, quantization_recipes

def getModelAndTokenizer(model_id):
    model = SparseAutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def preprocess(example, tokenizer):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }

def tokenize(sample,tokenizer):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

def getDataset(dataset_id, dataset_split, model_tokenizer):
    ds = load_dataset(dataset_id, dataset_split)
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))
    ds = ds.map(preprocess, fn_kwargs={"tokenizer": model_tokenizer})
    ds = ds.map(tokenize, fn_kwargs={"tokenizer": model_tokenizer}, remove_columns=ds.column_names)
    return ds

def getQuantizationRecipe(quantize_method):
    return quantization_recipes[quantize_method]["recipe"], quantization_recipes[quantize_method]["needs_calibration"]

def applyQuantization(model, tokenizer, quantize_method):
    recipe, needs_calibration = getQuantizationRecipe(quantize_method)
    if needs_calibration:
        ds = getDataset(DATASET_ID, DATASET_SPLIT, tokenizer)
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        )
    else:
        oneshot(model=model, recipe=recipe)
    
    SAVE_DIR = MODEL_ID.split("/")[1] + f"-{quantize_method}"
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    


if __name__ == "__main__":
    model, tokenizer = getModelAndTokenizer(MODEL_ID)
    quantize_method = "FP8_DYNAMIC"
    applyQuantization(model, tokenizer, quantize_method)


# # Confirm generations of the quantized model look sane.
# print("========== SAMPLE GENERATION ==============")
# input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
# output = model.generate(input_ids, max_new_tokens=20)
# print(tokenizer.decode(output[0]))
# print("==========================================")

