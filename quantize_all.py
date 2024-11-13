import subprocess
from config import quantization_recipes

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
    else:
        print(f"Error in quantization for method: {quantize_method}")
        print(result.stderr)

if __name__ == "__main__":
    # Iterate over all available quantization methods and run them one by one
    for method in quantization_recipes.keys():
        print(f"Running quantization for method: {method}")
        run_quantization_script(method)
        # break
        