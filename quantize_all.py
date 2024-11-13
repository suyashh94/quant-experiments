import subprocess
from config import quantization_recipes
import os

def run_quantization_script(quantize_method):
    # Command to activate Conda environment and run the script with the specified quantization method
    command = [
        "conda", "run", "--no-capture-output", "-n", "ml-gpu",
        "python", "quantize.py", "--quantize_method", quantize_method
    ]
    
    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Check if the command was executed successfully
    if result.returncode == 0:
        print(f"Successfully executed quantization for method: {quantize_method}")
        print(result.stdout)
        error_string = "No errors occurred"
    else:
        print(f"Error in quantization for method: {quantize_method}")
        print(result.stderr)
        error_string = result.stderr
    
    return error_string
    
    

if __name__ == "__main__":
    # Iterate over all available quantization methods and run them one by one
    for method in quantization_recipes.keys():
        print(f"Running quantization for method: {method}")
        error_string = run_quantization_script(method)
        # break
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True) 
        with open(f"{logs_dir}/{method}.txt", "w") as f:
            f.write(error_string)
        