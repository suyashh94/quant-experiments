import os 
import subprocess

def get_model_folders():
    # list all folders starting with "Phi-3.5-mini-instruct" in the current directory
    return [f for f in os.listdir() if f.startswith("Phi-3.5-mini-instruct")]

def get_quantization_method(folder_name):
    # extract the quantization method from the folder name
    return folder_name.split("-")[-1]

def evaluate_model(model_folder):
    quantize_method = get_quantization_method(model_folder)
    
    bash_command = f'''lm_eval --model vllm \
  --model_args pretrained='./{model_folder}',add_bos_token=true \
  --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,gsm8k \
  --num_fewshot 5 \
  --limit 500 \
  --batch_size 2 \
  --output_path 'eval-results/{quantize_method}'
  '''
  
    conda_env = "ml-gpu"
    command = [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "bash", "-c", bash_command
    ]
    
    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)
    # return result
    error_string = "No errors occurred"
    # Check if the command was executed successfully
    if result.returncode == 0:
        print(f"Successfully evaluated model: {model_folder}")
        print(result.stdout)
    else:
        print(f"Error in evaluating model: {model_folder}")
        print(result.stderr)
        error_string = result.stderr
    
    return error_string
    
if __name__ == "__main__":
    model_folders = get_model_folders()
    for model_folder in model_folders:
        print(f"Evaluating model: {model_folder}")
        error_string = evaluate_model(model_folder)
        # break
        # break after evaluating the first model for testing purposes
        # remove the break statement to evaluate all models
        
        logs_dir = 'eval-logs'
        os.makedirs(logs_dir, exist_ok=True)
        with open(f"{logs_dir}/{model_folder}.txt", "w") as f:
            f.write(error_string)
        