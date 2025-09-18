# █░█░░░░▒▓█▓▒░░▒▓█▓▒░░ ONNX BLUEPRINT GENERATOR ░░▒▓█▓▒░░▒▓█▓▒░░█░█░█
#
# This script inspects a compiled .onnx model and generates a human-readable
# blueprint of its internal learned parameters (weights and biases), known as
# "initializers".
#
# Its purpose is to be compared with the blueprint generated from the original
# PyTorch state dictionary to ensure that the model architecture and parameter
# shapes have not been corrupted or altered during the conversion process.

import os
import sys
import onnx

# --- CONFIGURATION ---
MODEL_FILENAME = ".\models\inswapper_128.onnx"
OUTPUT_FILENAME = "blueprint_inswapper.txt"

# --- SCRIPT LOGIC ---
def generate_onnx_blueprint():
    """
    Loads an ONNX model, inspects its graph initializers, and saves a
    structured blueprint to a text file for analysis.
    """
    # Assume this script is in a subdirectory like 'src/tools'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Go up one level to project root
    model_path = os.path.join(project_root, MODEL_FILENAME)
    output_path = os.path.join(script_dir, OUTPUT_FILENAME)

    if not os.path.exists(model_path):
        print(f"Error: ONNX model file not found at '{model_path}'")
        sys.exit(1)

    print(f"Loading ONNX model from '{model_path}'...")

    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"Error loading ONNX file: {e}")
        print("The file may be corrupted or not a valid ONNX model.")
        sys.exit(1)

    print(f"Generating ONNX blueprint and saving to '{output_path}'...")

    # Extract all initializers (trained parameters) from the model graph
    initializers = {init.name: init for init in model.graph.initializer}

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL BLUEPRINT FROM ONNX INITIALIZERS\n")
        f.write("="*80 + "\n\n")

        # Section 1: All Initializer Keys (Parameter Names)
        f.write("--- All Initializer Keys ---\n")
        for i, key in enumerate(sorted(initializers.keys())):
            f.write(f"[{i+1:03d}] {key}\n")
        f.write("\n")

        # Section 2: Detailed Initializer Information
        f.write("--- Detailed Initializer Information ---\n")

        # Helper to convert ONNX data type enum to a string
        def get_data_type_str(onnx_data_type):
            return onnx.TensorProto.DataType.Name(onnx_data_type)

        for name in sorted(initializers.keys()):
            init = initializers[name]
            shape = list(init.dims)
            data_type = get_data_type_str(init.data_type)

            f.write(f"\n--- Initializer: {name} ---\n")
            f.write(f"  Shape: {shape}\n")
            f.write(f"  Data Type: {data_type}\n")

    print(f"Successfully generated ONNX blueprint at '{output_path}'")


if __name__ == "__main__":
    generate_onnx_blueprint()