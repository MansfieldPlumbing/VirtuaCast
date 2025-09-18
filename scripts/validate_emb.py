import os
import struct
import numpy as np

# --- Configuration ---
# The expected number of floating-point values in each embedding file.
EXPECTED_FLOATS = 512
# The expected file size in bytes (512 floats * 4 bytes/float).
EXPECTED_SIZE_BYTES = EXPECTED_FLOATS * 4
# The L2 norm of the embedding vector should be 1.0. This is the allowed tolerance.
NORM_TOLERANCE = 0.01

# --- ANSI Color Codes for Prettier Output ---
C_GREEN = '\033[92m'
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_BLUE = '\033[94m'
C_RESET = '\033[0m'

def validate_embeddings():
    """
    Scans the 'Sources' directory for .emb files and validates their integrity.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sources_dir = os.path.join(script_dir, '..', 'Sources')
    
    print(f"\n{C_BLUE}--- VirtuaCast Embedding Validator ---{C_RESET}")
    print(f"Scanning for .emb files in: {os.path.abspath(sources_dir)}\n")

    if not os.path.isdir(sources_dir):
        print(f"{C_RED}Error: 'Sources' directory not found at the expected location.{C_RESET}")
        return

    emb_files = [f for f in os.listdir(sources_dir) if f.endswith('.emb')]

    if not emb_files:
        print(f"{C_YELLOW}No .emb files found in the 'Sources' directory.{C_RESET}")
        return

    valid_count = 0
    invalid_count = 0

    for filename in sorted(emb_files):
        file_path = os.path.join(sources_dir, filename)
        errors = []

        # --- Check 1: File Size ---
        actual_size = os.path.getsize(file_path)
        if actual_size != EXPECTED_SIZE_BYTES:
            errors.append(f"Invalid file size: {actual_size} bytes (expected {EXPECTED_SIZE_BYTES})")

        # --- Check 2: L2-Normalization ---
        if not errors:
            try:
                with open(file_path, 'rb') as f:
                    # Read all 512 floats from the binary file
                    content = f.read()
                    embedding = struct.unpack(f'<{EXPECTED_FLOATS}f', content)
                    
                    # Use NumPy to calculate the L2 norm
                    vector = np.array(embedding, dtype=np.float32)
                    norm = np.linalg.norm(vector)

                    if abs(norm - 1.0) > NORM_TOLERANCE:
                        errors.append(f"Not L2-normalized. Norm = {norm:.6f}")

            except (struct.error, EOFError):
                errors.append("File is corrupt and could not be read as 512 floats.")
            except Exception as e:
                errors.append(f"An unexpected error occurred: {e}")

        # --- Print Result ---
        if not errors:
            print(f"  {C_GREEN}[PASS] {filename}{C_RESET}")
            valid_count += 1
        else:
            print(f"  {C_RED}[FAIL] {filename}{C_RESET}")
            for error in errors:
                print(f"    - {C_YELLOW}{error}{C_RESET}")
            invalid_count += 1

    print("\n" + "="*40)
    print("Validation Summary:")
    print(f"  {C_GREEN}Valid files:   {valid_count}{C_RESET}")
    print(f"  {C_RED}Invalid files: {invalid_count}{C_RESET}")
    print("="*40)
    
    if invalid_count > 0:
        print(f"\n{C_YELLOW}Suggestion: Delete the {invalid_count} invalid .emb file(s).")
        print(f"VirtuaCast will automatically regenerate them the next time it runs.{C_RESET}")


if __name__ == "__main__":
    # This allows the script to be run directly from the command line.
    validate_embeddings()