import os
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

# --- ANSI Color Codes for Output ---
C_BLUE, C_GREEN, C_RED, C_RESET = '\033[94m', '\033[92m', '\033[91m', '\033[0m'

# --- CONFIGURATION ---
CONFIG = {
    "source_image_path": "../Sources/source.jpg", # CHANGE THIS TO YOUR IMAGE
    "model_directory": "../models",
    "output_dir": "./validation_output"
}
# ---------------------

def estimate_norm(lmk, image_size=112):
    """Replicates the scikit-image SimilarityTransform estimation using OpenCV."""
    arcface_dst = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
    ratio = float(image_size) / 112.0
    dst = arcface_dst * ratio
    tform = cv2.estimateAffinePartial2D(lmk.astype(np.float32), dst)[0]
    return tform

def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

# Replicating the FaceDetector class from your original script
class FaceDetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        input_shape = list(input_cfg.shape)
        if not isinstance(input_shape[2], int) or not isinstance(input_shape[3], int):
            input_shape[2], input_shape[3] = 640, 640
        self.input_size = tuple(input_shape[2:4][::-1])
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.fmc = 3
        
    def forward(self, img_blob):
        net_outs = self.session.run(self.output_names, {self.input_name: img_blob})
        input_height, input_width = img_blob.shape[2], img_blob.shape[3]
        scores_list, bboxes_list, kpss_list = [], [], []

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + self.fmc] * stride
            kps_preds = net_outs[idx + self.fmc * 2] * stride

            # Reshape the output to a consistent format (assuming BCHW)
            # Check if the output is already in the correct format before transposing
            if scores.ndim == 4 and scores.shape[1] > 1: # Check for BCHW or similar
                 scores = scores.transpose((0, 2, 3, 1))
                 bbox_preds = bbox_preds.transpose((0, 2, 3, 1))
                 kps_preds = kps_preds.transpose((0, 2, 3, 1))
        
            scores = scores.reshape(-1)
            # Reshape other tensors as well
            bbox_preds = bbox_preds.reshape(-1, 4)
            kps_preds = kps_preds.reshape(-1, 10)

            height, width = input_height // stride, input_width // stride

            anchor_centers = self.center_cache.get((height, width, stride))
            if anchor_centers is None:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                self.center_cache[(height, width, stride)] = anchor_centers

            pos_inds = np.where(scores >= self.det_thresh)[0]

            bboxes = distance2bbox(anchor_centers[pos_inds], bbox_preds[pos_inds])
            kpss = distance2kps(anchor_centers[pos_inds], kps_preds[pos_inds])

            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes)
            kpss_list.append(kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img):
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(self.input_size[1]) / self.input_size[0]
        new_height = self.input_size[1] if im_ratio > model_ratio else int(self.input_size[0] * im_ratio)
        new_width = int(self.input_size[1] / im_ratio) if im_ratio > model_ratio else self.input_size[0]
        det_scale = float(new_height) / img.shape[0]

        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
    
        blob = cv2.dnn.blobFromImage(det_img, 1.0/128.0, self.input_size, (127.5, 127.5, 127.5), swapRB=True)
        scores_list, bboxes_list, kpss_list = self.forward(blob)

        # --- Add this filtering step ---
        # Combine the lists and filter out empty arrays in one go.
        combined_list = [(s, b, k) for s, b, k in zip(scores_list, bboxes_list, kpss_list) if s.size > 0]
    
        # If no faces were detected in any feature map, return an empty list.
        if not combined_list:
            print(f"{C_RED}No faces detected by the model above threshold. Double-check image.{C_RESET}")
            return []

        # Unzip the combined list back into separate lists.
        scores_list, bboxes_list, kpss_list = zip(*combined_list)
        # --- End filtering step ---
    
        scores = np.vstack(scores_list).ravel()
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale

        order = scores.argsort()[::-1]
        pre_det = np.hstack((bboxes, scores[:, np.newaxis]))[order, :]
        
        keep = cv2.dnn.NMSBoxes(pre_det[:,:4].tolist(), pre_det[:,4].tolist(), self.det_thresh, self.nms_thresh)
        if keep is None or len(keep) == 0: return []
    
        indices = keep.flatten()
        final_kpss = kpss[order][indices]
    
        return [{'landmarks': final_kpss[i].reshape(5, 2)} for i in range(len(indices))]

def visualize_embedding_and_face(embedding, aligned_face, config):
    """Generates a heatmap of the embedding and saves the aligned face."""
    # Ensure output directory exists
    os.makedirs(config['output_dir'], exist_ok=True)

    # 1. Visualize embedding as a heatmap
    embedding_reshaped = embedding.reshape((16, 32))
    plt.figure(figsize=(10, 5))
    plt.imshow(embedding_reshaped, cmap='viridis', aspect='auto')
    plt.title('Embedding Heatmap')
    plt.colorbar(label='Value')
    heatmap_path = os.path.join(config['output_dir'], "embedding_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"{C_GREEN}Embedding heatmap saved to '{heatmap_path}'{C_RESET}")

    # 2. Save the cropped, aligned face
    face_path = os.path.join(config['output_dir'], "aligned_face_input.png")
    cv2.imwrite(face_path, aligned_face)
    print(f"{C_GREEN}Aligned face image saved to '{face_path}'{C_RESET}")


def run_validation(config):
    print(f"\n{C_BLUE}--- Embedding Validation Script ---{C_RESET}")

    det_model_path = os.path.join(config['model_directory'], "det_10g.onnx")
    rec_model_path = os.path.join(config['model_directory'], "w600k_r50.onnx")
    
    try:
        print("Loading ONNX models...")
        face_detector = FaceDetector(model_path=det_model_path)
        rec_session = ort.InferenceSession(rec_model_path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
        print(f"{C_GREEN}Models loaded successfully using: {ort.get_device()}{C_RESET}")
    except Exception as e:
        print(f"{C_RED}ERROR: Failed to load ONNX models. {e}{C_RESET}"); return

    print(f"Loading source image from '{config['source_image_path']}'...")
    source_img = cv2.imread(config['source_image_path'])
    if source_img is None:
        print(f"{C_RED}ERROR: Failed to load source image. Check the path in CONFIG.{C_RESET}"); return

    print("Detecting face in source image...")
    source_faces = face_detector.detect(source_img)
    if not source_faces:
        print(f"{C_RED}ERROR: No face found in source image.{C_RESET}"); return
    
    source_face = source_faces[0]
    
    # Generate the aligned face image for the recognition model
    M_align_src = estimate_norm(source_face['landmarks'], image_size=112)
    aligned_source_face = cv2.warpAffine(source_img, M_align_src, (112, 112), borderValue=0.0)
    
    # Generate the embedding
    rec_blob = cv2.dnn.blobFromImage(aligned_source_face, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    source_embedding = rec_session.run(None, {'input.1': rec_blob})[0]
    source_embedding /= np.linalg.norm(source_embedding)
    
    # Visualize and save results
    visualize_embedding_and_face(source_embedding, aligned_source_face, config)
    
    print(f"\n{C_GREEN}Embedding validation complete. Check the '{config['output_dir']}' folder.{C_RESET}")
    print("Embedding statistics:")
    print(f"  Shape: {source_embedding.shape}")
    print(f"  Total size (bytes): {source_embedding.size * source_embedding.itemsize}")
    print(f"  Sum of squared values (should be ~1.0): {np.sum(np.square(source_embedding))}")
    print(f"  Min value: {np.min(source_embedding)}")
    print(f"  Max value: {np.max(source_embedding)}")


if __name__ == '__main__':
    run_validation(CONFIG)
