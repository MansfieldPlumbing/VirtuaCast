import os
import numpy as np

# --- ANSI Color Codes for Output ---
C_BLUE, C_GREEN, C_RED, C_YELLOW, C_RESET = '\033[94m', '\033[92m', '\033[91m', '\033[93m', '\033[0m'

def validate_embedding_file(file_path):
    """
    Validates a 512-dimensional embedding file by checking its size and normalization.

    Args:
        file_path (str): The path to the binary .emb file.
    """
    if not os.path.exists(file_path):
        print(f"{C_RED}ERROR: File not found at '{file_path}'{C_RESET}")
        return

    expected_size_bytes = 512 * 4  # 512 floats * 4 bytes/float
    
    # Check the file size
    actual_size_bytes = os.path.getsize(file_path)
    if actual_size_bytes != expected_size_bytes:
        print(f"{C_RED}ERROR: File size mismatch.{C_RESET}")
        print(f"  Expected: {expected_size_bytes} bytes (512 floats)")
        print(f"  Actual:   {actual_size_bytes} bytes")
        return

    # Read the raw binary data into a NumPy array of 32-bit floats
    embedding = np.fromfile(file_path, dtype=np.float32)

    # Check the shape of the resulting array
    if embedding.shape != (512,):
        print(f"{C_RED}ERROR: Array shape mismatch.{C_RESET}")
        print(f"  Expected: (512,)")
        print(f"  Actual:   {embedding.shape}")
        return

    # Calculate the magnitude (L2 norm)
    magnitude = np.linalg.norm(embedding)

    # Check if the embedding is properly normalized
    is_normalized = np.isclose(magnitude, 1.0, atol=1e-6)

    print(f"\n{C_BLUE}--- Embedding File Validation Report ---{C_RESET}")
    print(f"{C_YELLOW}File: {file_path}{C_RESET}")
    print(f"  Size: {actual_size_bytes} bytes ({actual_size_bytes/1024:.2f} KB)")
    print(f"  Shape: {embedding.shape}")
    print(f"  Magnitude (L2 Norm): {magnitude:.8f}")
    print(f"  Min value: {np.min(embedding):.8f}")
    print(f"  Max value: {np.max(embedding):.8f}")
    print(f"  Number of zero values: {np.sum(embedding == 0)}")
    
    if is_normalized:
        print(f"\n{C_GREEN}Validation SUCCESS: Embedding is correctly normalized.{C_RESET}")
    else:
        print(f"\n{C_RED}Validation FAILED: Embedding is NOT correctly normalized.{C_RESET}")
        print(f"  The magnitude should be very close to 1.0.")

if __name__ == '__main__':
    emb_file_path = os.path.join(os.path.dirname(__file__), "test.emb")
    validate_embedding_file(emb_file_path)

