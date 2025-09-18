import numpy as np
import matplotlib.pyplot as plt
import os

# --- ANSI Color Codes for Output ---
C_BLUE, C_GREEN, C_RED, C_RESET = '\033[94m', '\033[92m', '\033[91m', '\033[0m'

def visualize_embedding():
    """
    Reads the 512-float embedding, pads it to the next perfect square (529),
    reshapes it into a 23x23 grid, and saves it as a JPG heatmap.
    """
    input_file = '../DEBUG_01a_Source_Embedding.txt'
    output_file = '../DEBUG_01b_Source_Embedding_Heatmap.jpg'
    
    print(f"\n{C_BLUE}--- Generating SQUARE Embedding Heatmap ---{C_RESET}")

    if not os.path.exists(input_file):
        print(f"{C_RED}ERROR: Input file not found: '{input_file}'")
        print("Please run a debug trace in VirtuaCast first to generate the embedding file.{C_RESET}")
        return

    try:
        embedding = np.loadtxt(input_file, dtype=np.float32)

        if embedding.shape != (512,):
            print(f"{C_RED}ERROR: Embedding file has incorrect dimensions. Expected 512 floats, found {embedding.size}.{C_RESET}")
            return
            
        print(f"Successfully loaded embedding with {embedding.size} float values.")

        # Pad the 512-element vector to 529 (23*23) with zeros.
        padded_embedding = np.zeros(23 * 23, dtype=np.float32)
        padded_embedding[:512] = embedding
        
        # Reshape the padded array into a 23x23 square grid.
        heatmap_grid = padded_embedding.reshape((23, 23))

        # Use matplotlib to save the square grid as a colormapped image.
        plt.imsave(output_file, heatmap_grid, cmap='viridis', format='jpg')

        print(f"{C_GREEN}Successfully generated and saved SQUARE heatmap to '{output_file}'{C_RESET}")

    except Exception as e:
        print(f"{C_RED}An error occurred: {e}{C_RESET}")

if __name__ == '__main__':
    visualize_embedding()