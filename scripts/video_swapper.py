"""
VirtuaCast Video Swapper (Definitive Version with Preview and Advanced Mask/Crop Control)

This script prompts the user to select a source face image and a target video file.
It then applies a face swap to each frame of the video and saves the result as a
new MP4 file on the user's Desktop.

This version includes:
- A completely rewritten FaceDetector.forward method to handle 2D model output.
- A live preview window during processing (close by pressing 'q').
- Configurable 'crop_area_expansion_ratio' to expand the final square crop area (symmetric).
- Configurable 'chin_extra_padding_ratio' to explicitly push the bottom of the face
  bounding box down, ensuring the chin is initially captured.
- Configurable 'mask_scale_factor' to adjust the final blending mask size.

Crucially, this version uses the FULL bounding box returned by the face detector
as the base for expansion, explicitly ensures chin inclusion, and then performs
a symmetric square expansion. Visual debug bounding boxes are drawn in the preview
and detailed coordinates are printed to the console for verification.

Instructions:
  1. Ensure required libraries are installed:
     pip install numpy opencv-python onnx onnxruntime tqdm
  2. Run the script from the 'scripts' directory:
     > python video_swapper.py
  3. Follow the prompts to select your source image and target video.
  4. Adjust `crop_area_expansion_ratio`, `chin_extra_padding_ratio`, and
     `mask_scale_factor` in the CONFIG section to experiment with crop and blend sizes.
"""

import os
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog

# --- ANSI Color Codes for Output ---
C_BLUE, C_GREEN, C_RED, C_RESET = '\033[94m', '\033[92m', '\033[91m', '\033[0m'

# --- Library Imports with Error Handling ---
try:
    import cv2
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper
    from tqdm import tqdm
except ImportError as e:
    print(f"{C_RED}FATAL: A critical library is missing: {e}{C_RESET}")
    print(f"{C_RED}Please run 'pip install numpy opencv-python onnx onnxruntime tqdm' to fix this.{C_RESET}")
    sys.exit(1)

# --- CONFIGURATION ---
CONFIG = {
    "output_filename": "video_swap_result.mp4",
    "crop_area_expansion_ratio": 0.3, # Factor for symmetric expansion of the face region (e.g., 0.3 for 30% larger).
                                      # This defines the final square crop size for the 128x128 model input.
    "chin_extra_padding_ratio": 0.15, # NEW: Explicitly extends the bottom of the detected face bbox by this ratio of its height.
                                     # Guarantees chin inclusion. Adjust this first to see more chin.
    "mask_scale_factor": 1.0          # Factor to increase/decrease the final blend mask area (e.g., 1.3 for 30% larger).
                                     # This affects the final blending on the output.
}
# ---------------------

# --- Core Algorithm Functions ---
def estimate_norm(lmk, image_size=112):
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

class FaceDetector:
    def __init__(self, model_path):
        # Ensure DirectML is attempted first
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
            
            height, width = input_height // stride, input_width // stride
            
            anchor_centers = self.center_cache.get((height, width, stride))
            if anchor_centers is None:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                self.center_cache[(height, width, stride)] = anchor_centers

            if scores.ndim == 4:
                scores = scores.transpose((0, 2, 3, 1)).reshape(-1)
            else:
                scores = scores.reshape(-1)

            if bbox_preds.ndim == 4:
                bbox_preds = bbox_preds.transpose((0, 2, 3, 1)).reshape(-1, 4)
            else:
                bbox_preds = bbox_preds.reshape(-1, 4)

            if kps_preds.ndim == 4:
                kps_preds = kps_preds.transpose((0, 2, 3, 1)).reshape(-1, 10)
            else:
                kps_preds = kps_preds.reshape(-1, 10)

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
        
        scores_list_raw, bboxes_list_raw, kpss_list_raw = self.forward(blob)
        
        filtered_scores, filtered_bboxes, filtered_kpss = [], [], []
        for s, b, k in zip(scores_list_raw, bboxes_list_raw, kpss_list_raw):
            if s.size > 0:
                filtered_scores.append(s)
                filtered_bboxes.append(b)
                filtered_kpss.append(k)
        
        if not filtered_scores:
            return []

        scores = np.concatenate(filtered_scores)
        bboxes = np.vstack(filtered_bboxes) / det_scale # These are the actual bounding boxes
        kpss = np.vstack(filtered_kpss) / det_scale

        order = scores.argsort()[::-1]
        pre_det = np.hstack((bboxes, scores[:, np.newaxis]))[order, :]
        
        keep_result = cv2.dnn.NMSBoxes(pre_det[:,:4].tolist(), pre_det[:,4].tolist(), self.det_thresh, self.nms_thresh)
        keep = keep_result.flatten() if keep_result is not None else []
        
        if len(keep) == 0: return []
        
        final_bboxes = pre_det[:,:4][order][keep] # Extract the final chosen bounding boxes (x1, y1, x2, y2)
        final_kpss = kpss[order][keep]
        
        # Return both the bbox and landmarks
        return [{'bbox': final_bboxes[i], 'landmarks': final_kpss[i].reshape(5, 2)} for i in range(len(keep))]

def extract_emap_from_model(model_path):
    try:
        model = onnx.load(model_path)
        for initializer in model.graph.initializer:
            if initializer.name == 'buff2fs':
                arr = numpy_helper.to_array(initializer)
                if arr.shape == (512, 512):
                    return arr.astype(np.float32)
    except Exception as e:
        print(f"{C_RED}Could not read ONNX model for EMAP extraction: {e}{C_RESET}")
    return None

def process_frame(target_img, source_latent, face_detector, swap_session, config):
    target_faces = face_detector.detect(target_img)
    if not target_faces:
        return target_img 
    
    target_face = target_faces[0]
    
    # 1. Start with the detected bounding box as the base
    initial_bbox = target_face['bbox'].astype(float) # Use float for precise calculations
    initial_x1, initial_y1, initial_x2, initial_y2 = initial_bbox[0], initial_bbox[1], initial_bbox[2], initial_bbox[3]

    # Calculate current face width and height
    face_width = initial_x2 - initial_x1
    face_height = initial_y2 - initial_y1

    # 2. Apply chin_extra_padding_ratio to explicitly extend the bottom
    chin_extra_padding_ratio = config.get("chin_extra_padding_ratio", 0.0)
    extra_chin_padding = face_height * chin_extra_padding_ratio
    
    # Calculate the new bottom (y2) after adding chin padding
    padded_y2 = initial_y2 + extra_chin_padding

    # Calculate the center of the *initial* bounding box (before any padding or symmetric expansion)
    center_x = (initial_x1 + initial_x2) / 2.0
    center_y = (initial_y1 + initial_y2) / 2.0

    # 3. Determine the dimensions for the square crop based on the chin-padded height
    # We want a square crop that, at minimum, covers the width of the face
    # and the new (chin-padded) height.
    effective_height_for_square = padded_y2 - initial_y1 # height from initial y1 to padded y2

    # The side length of our square crop should be based on the maximum of face_width
    # and this effective_height_for_square, then expanded.
    base_side_length = max(face_width, effective_height_for_square)

    crop_area_expansion_ratio = config.get("crop_area_expansion_ratio", 0.0)
    desired_crop_side = base_side_length * (1.0 + crop_area_expansion_ratio)

    # 4. Calculate the coordinates of the final expanded square bounding box
    # We will center this square around a point that is vertically shifted to incorporate the chin padding
    # The center for this square should ideally be the original center_x, but shifted down to account for padded_y2
    
    # Calculate the new center_y for the expanded square, such that its bottom edge
    # extends sufficiently to cover the chin padding.
    # A simple way: keep center_x, and calculate center_y from the middle of the *desired_crop_side* height
    # placed such that its bottom is at (padded_y2 + some margin, if needed).
    # Or, we can center it on the original face center, and then shift the whole square downwards.
    # Let's try centering it on original (center_x, initial_y1 + desired_crop_side / 2.0)
    
    final_center_x = center_x
    # The y-center for the square should ensure it extends from top to bottom
    # with the extra chin padding.
    # A robust way: compute the expanded box first, then clamp.
    
    expanded_x1_float = final_center_x - desired_crop_side / 2.0
    expanded_x2_float = final_center_x + desired_crop_side / 2.0
    
    # Calculate y-coordinates such that the bottom aligns with padded_y2, and the total height is desired_crop_side
    expanded_y2_float = padded_y2 + (desired_crop_side * 0.1) # Add a small buffer below chin for aesthetic
    expanded_y1_float = expanded_y2_float - desired_crop_side

    # Ensure integer coordinates for drawing and cropping
    expanded_x1, expanded_y1, expanded_x2, expanded_y2 = \
        int(round(expanded_x1_float)), int(round(expanded_y1_float)), int(round(expanded_x2_float)), int(round(expanded_y2_float))

    # Clamping to image boundaries
    h_img, w_img, _ = target_img.shape
    expanded_x1 = max(0, expanded_x1)
    expanded_y1 = max(0, expanded_y1)
    expanded_x2 = min(w_img, expanded_x2)
    expanded_y2 = min(h_img, expanded_y2)

    # Ensure valid dimensions after clamping (at least 2x2 pixels for affine transform stability)
    if expanded_x2 <= expanded_x1 + 1 or expanded_y2 <= expanded_y1 + 1:
        print(f"{C_RED}WARNING: Computed expanded box is invalid. Falling back to default crop or original frame.{C_RESET}")
        center_x_fallback = int(round(center_x))
        center_y_fallback = int(round(center_y))
        expanded_x1 = max(0, center_x_fallback - 64)
        expanded_y1 = max(0, center_y_fallback - 64)
        expanded_x2 = min(w_img, center_x_fallback + 64)
        expanded_y2 = min(h_img, center_y_fallback + 64)
        if expanded_x2 <= expanded_x1 + 1 or expanded_y2 <= expanded_y1 + 1:
            return target_img # Still invalid, return original frame

    # --- Debug printing ---
    print(f"\n--- Frame Debug ---")
    print(f"Initial bbox: ({int(initial_bbox[0])},{int(initial_bbox[1])},{int(initial_bbox[2])},{int(initial_bbox[3])})")
    print(f"Face dimensions (initial): W={face_width:.2f}, H={face_height:.2f}")
    print(f"Chin padding: {extra_chin_padding:.2f} pixels (ratio={chin_extra_padding_ratio:.2f})")
    print(f"Padded y2 (bottom after chin pad): {padded_y2:.2f}")
    print(f"Effective height for square (after chin pad): {effective_height_for_square:.2f}")
    print(f"Base side length for square crop: {base_side_length:.2f}")
    print(f"Crop area expansion ratio: {crop_area_expansion_ratio:.2f}")
    print(f"Desired final square crop side length: {desired_crop_side:.2f}")
    print(f"Final expanded bbox (x1,y1,x2,y2): ({expanded_x1},{expanded_y1},{expanded_x2},{expanded_y2})")
    print(f"-------------------")
    # --- End Debug printing ---

    # 5. Define source points (corners of the final expanded bbox in the original image)
    src_pts = np.array([
        [expanded_x1, expanded_y1],          # Top-left
        [expanded_x2, expanded_y1],          # Top-right
        [expanded_x1, expanded_y2]           # Bottom-left
    ], dtype=np.float32)

    # Define destination points (corners of the 128x128 canvas for the model input)
    dst_pts = np.array([
        [0, 0],
        [128, 0],
        [0, 128]
    ], dtype=np.float32)

    # Get the affine transform to map the expanded bbox region to 128x128
    M_to_128 = cv2.getAffineTransform(src_pts, dst_pts)
    
    # 6. Warp the target_img to get the 128x128 input for the swapper
    aligned_target_face = cv2.warpAffine(target_img, M_to_128, (128, 128), borderValue=0.0)
    
    swap_blob = cv2.dnn.blobFromImage(aligned_target_face, 1.0/255.0, (128, 128), (0.0, 0.0, 0.0), swapRB=True)
    swapped_face_raw = swap_session.run(None, {'target': swap_blob, 'source': source_latent})[0][0]
    swapped_face_img = np.clip(255 * swapped_face_raw.transpose((1, 2, 0)), 0, 255).astype(np.uint8)
    swapped_face_img = cv2.cvtColor(swapped_face_img, cv2.COLOR_RGB2BGR)
    
    try:
        # 7. Invert the affine transform to get the paste matrix (maps 128x128 back to the expanded bbox region)
        M_paste = cv2.invertAffineTransform(M_to_128)
    except cv2.error:
        return target_img
    
    # --- Apply mask_scale_factor to M_paste (as before) ---
    mask_scale_factor = config.get("mask_scale_factor", 1.0) 
    
    if mask_scale_factor != 1.0:
        # Calculate the center of the *final expanded* bounding box for scaling
        bbox_center_x = (expanded_x1 + expanded_x2) / 2
        bbox_center_y = (expanded_y1 + expanded_y2) / 2

        scale_matrix = cv2.getRotationMatrix2D(
            (bbox_center_x, bbox_center_y),
            0, # No rotation
            mask_scale_factor
        )

        M_paste_3x3 = np.vstack((M_paste, [0, 0, 1]))
        scale_matrix_3x3 = np.vstack((scale_matrix, [0, 0, 1]))
        M_paste = (scale_matrix_3x3 @ M_paste_3x3)[:2, :] # Combine and convert back to 2x3
    # --- End mask_scale_factor application ---

    # Warp the 128x128 swapped face back into the original image's coordinate space
    full_frame_swapped_face = cv2.warpAffine(swapped_face_img, M_paste, (w_img, h_img))
    
    # Generate a simple 128x128 mask and warp it using the same M_paste matrix
    mask_template = np.full((128, 128), 255, dtype=np.uint8)
    full_frame_mask = cv2.warpAffine(mask_template, M_paste, (w_img, h_img))
    
    full_frame_mask_blurred = cv2.GaussianBlur(full_frame_mask, (21, 21), 0)
    mask_float = (full_frame_mask_blurred.astype(np.float32) / 255.0)[:, :, np.newaxis]
    
    final_result = (full_frame_swapped_face * mask_float + target_img * (1 - mask_float)).astype(np.uint8)

    # --- Visual Debugging: Draw bounding boxes on the output frame ---
    # Draw initial (chin-padded) bounding box in green
    cv2.rectangle(final_result, (initial_x1, initial_y1), (initial_x2, int(round(padded_y2))), (0, 255, 0), 2) # Use padded_y2 for green
    # Draw final expanded bounding box in blue
    cv2.rectangle(final_result, (expanded_x1, expanded_y1), (expanded_x2, expanded_y2), (255, 0, 0), 2)
    # --- End Visual Debugging ---

    return final_result

def get_file_path_from_dialog(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_path

def run_video_pipeline(config):
    print(f"\n{C_BLUE}--- VirtuaCast Video Swapper ---{C_RESET}")

    source_image_path = get_file_path_from_dialog(
        title="STEP 1: Select the SOURCE FACE IMAGE",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )
    if not source_image_path:
        print(f"{C_RED}No source image selected. Exiting.{C_RESET}"); sys.exit(0)
    print(f"Selected source image: {C_BLUE}{source_image_path}{C_RESET}")

    target_video_path = get_file_path_from_dialog(
        title="STEP 2: Select the TARGET VIDEO",
        filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")]
    )
    if not target_video_path:
        print(f"{C_RED}No target video selected. Exiting.{C_RESET}"); sys.exit(0)
    print(f"Selected target video: {C_BLUE}{target_video_path}{C_RESET}")

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_model_dir = os.path.join(script_dir, "..", "models")
        det_model_path = os.path.join(absolute_model_dir, "det_10g.onnx")
        rec_model_path = os.path.join(absolute_model_dir, "w600k_r50.onnx")
        swap_model_path = os.path.join(absolute_model_dir, "inswapper_128.onnx")
        for path in [det_model_path, rec_model_path, swap_model_path]:
            if not os.path.exists(path):
                print(f"{C_RED}FATAL: Model file not found at: {path}{C_RESET}")
                sys.exit(1)
    except Exception as e:
        print(f"{C_RED}FATAL: Error constructing model paths: {e}{C_RESET}")
        sys.exit(1)

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    output_video_path = os.path.join(desktop_path, config['output_filename'])

    try:
        print("\nLoading ONNX models...")
        face_detector = FaceDetector(model_path=det_model_path)
        rec_session = ort.InferenceSession(rec_model_path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
        swap_session = ort.InferenceSession(swap_model_path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
        
        det_provider = face_detector.session.get_providers()[0]
        rec_provider = rec_session.get_providers()[0]
        swap_provider = swap_session.get_providers()[0]

        print(f"{C_GREEN}Models loaded successfully using:{C_RESET}")
        print(f"  Face Detector: {C_GREEN}{det_provider}{C_RESET}")
        print(f"  Recognition:   {C_GREEN}{rec_provider}{C_RESET}")
        print(f"  Swapper:       {C_GREEN}{swap_provider}{C_RESET}")

    except Exception as e:
        print(f"{C_RED}FATAL: Failed to load ONNX models. Error: {e}{C_RESET}")
        sys.exit(1)

    emap = extract_emap_from_model(swap_model_path)
    if emap is None:
        print(f"{C_RED}FATAL: Could not find EMAP matrix in '{swap_model_path}'.{C_RESET}"); sys.exit(1)

    print("\nProcessing source image...")
    source_img = cv2.imread(source_image_path)
    if source_img is None:
        print(f"{C_RED}FATAL: OpenCV failed to load source image: '{source_image_path}'{C_RESET}"); sys.exit(1)

    source_faces = face_detector.detect(source_img)
    if not source_faces:
        print(f"{C_RED}FATAL: No face found in source image.{C_RESET}"); sys.exit(1)

    source_face = source_faces[0]
    # For source face, we still use the standard alignment based on landmarks for embedding extraction
    # as the recognition model is trained on these specific alignments.
    M_align_src = estimate_norm(source_face['landmarks'], image_size=112)
    aligned_source_face = cv2.warpAffine(source_img, M_align_src, (112, 112), borderValue=0.0)
    rec_blob = cv2.dnn.blobFromImage(aligned_source_face, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    source_embedding = rec_session.run(None, {'input.1': rec_blob})[0]
    source_embedding /= np.linalg.norm(source_embedding)

    source_latent = np.dot(source_embedding, emap)
    source_latent /= np.linalg.norm(source_latent)
    print(f"{C_GREEN}Source face processed successfully.{C_RESET}")
    print(f"Crop area expansion ratio set to: {C_BLUE}{config['crop_area_expansion_ratio']:.2f}{C_RESET}")
    print(f"Chin extra padding ratio set to: {C_BLUE}{config['chin_extra_padding_ratio']:.2f}{C_RESET}")
    print(f"Mask blend scale factor set to: {C_BLUE}{config['mask_scale_factor']:.2f}{C_RESET}")


    print(f"\nOpening and processing target video...")
    cap = cv2.VideoCapture(target_video_path)
    if not cap.isOpened():
        print(f"{C_RED}FATAL: Could not open video file.{C_RESET}"); sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    cv2.namedWindow("Video Swap Preview", cv2.WINDOW_NORMAL) 

    for _ in tqdm(range(total_frames), desc="Swapping Frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame, source_latent, face_detector, swap_session, config)
        
        cv2.imshow("Video Swap Preview", processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"\n{C_BLUE}Preview closed by user. Finishing video saving...{C_RESET}")
            break

        writer.write(processed_frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n{C_GREEN}--- Video Processing Complete ---{C_RESET}")
    print(f"Swapped video saved to: {C_BLUE}{output_video_path}{C_RESET}")

if __name__ == '__main__':
    run_video_pipeline(CONFIG)