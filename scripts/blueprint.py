import torch
import os
import sys

# --- CONFIGURATION ---
MODEL_FILENAME = "student_128.pth"
OUTPUT_FILENAME = "blueprint.txt"

# --- SCRIPT LOGIC ---
def generate_blueprint():
    """
    Loads the model state dictionary from a .pth file and generates
    a human-readable blueprint of its layers and parameters.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    model_path = os.path.join(project_root, 'models', MODEL_FILENAME)
    output_path = os.path.join(script_dir, OUTPUT_FILENAME)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        sys.exit(1)

    print(f"Loading state dictionary from '{model_path}'...")
    
    try:
        # Load the checkpoint, mapping to CPU to avoid CUDA errors
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading model file: {e}")
        print("The file may be corrupted or not a valid PyTorch state dictionary.")
        sys.exit(1)

    # If the .pth file is a full checkpoint, extract the model_state_dict
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    print(f"Generating blueprint and saving to '{output_path}'...")
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL BLUEPRINT FROM PYTORCH STATE DICTIONARY\n")
        f.write("="*80 + "\n\n")
        
        # Section 1: All Keys (Parameter Names)
        f.write("--- All Parameter Keys ---\n")
        for i, key in enumerate(sorted(state_dict.keys())):
            f.write(f"[{i+1:03d}] {key}\n")
        f.write("\n")
        
        # Section 2: Detailed Layer Information
        f.write("--- Detailed Layer Information ---\n")
        
        # Group keys by their parent module
        layer_groups = {}
        for key, tensor in state_dict.items():
            parts = key.split('.')
            if len(parts) > 1:
                module_name = '.'.join(parts[:-1])
                param_name = parts[-1]
            else:
                module_name = 'top_level'
                param_name = key
                
            if module_name not in layer_groups:
                layer_groups[module_name] = {}
            layer_groups[module_name][param_name] = tensor.shape
        
        # Write the grouped information
        for module_name, params in sorted(layer_groups.items()):
            f.write(f"\n--- Module: {module_name} ---\n")
            for param_name, shape in params.items():
                f.write(f"  {param_name}: {list(shape)}\n")
        
        print(f"Successfully generated blueprint at '{output_path}'")

if __name__ == "__main__":
    generate_blueprint()