import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# --- Configuration ---
# Modify these to test different source/target combinations.
CONFIG = {
    "source_image": "../Sources/Alyson Hannigan.jpg",
    "target_image": "test_data/target_frame.jpg",
    "model_dir": "../models",
    "output_dir": "PY_DEBUG"
}

# --- ANSI Color Codes ---
C_BLUE = '\033[94m'
C_GREEN = '\033[92m'
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_RESET = '\033[0m'

# --- Core Algorithm Replication ---
# This Python code is a direct port of the logic in your C++ Algorithms.h
# to ensure we are comparing the exact same operations.

def estimate_similarity_transform(src_pts, dst_pts):
    """
    Python/NumPy equivalent of the estimateSimilarityTransform function in Algorithms.h.
    """
    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)
    
    src_demean = src_pts - src_mean
    dst_demean = dst_pts - dst_mean
    
    H = src_demean.T @ dst_demean
    U, s, Vt = np.linalg.svd(H)
    
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        
    var_src = np.sum(src_demean**2) / len(src_pts)
    
    d = np.ones(H.shape[0])
    if np.linalg.det(H) < 0:
        d[-1] = -1
    
    scale = np.sum(s * d) / var_src
    
    t = dst_mean.T - scale * R @ src_mean.T
    
    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = scale * R
    M[:, 2] = t
    return M

def preprocess_detection_image(image, input_size=640):
    """
    Python equivalent of the letterboxing and normalization for the detection model.
    """
    h, w, _ = image.shape
    r = min(input_size / w, input_size / h)
    in_w, in_h = int(w * r), int(h * r)
    
    resized_img = cv2.resize(image, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    
    letterbox_img = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    letterbox_img[:in_h, :in_w, :] = resized_img
    
    blob = ((letterbox_img.astype(np.float32) - 127.5) / 128.0)
    blob = blob.transpose(2, 0, 1)
    return blob[np.newaxis, :, :, :], r


def postprocess_detection(outputs, r_scale, score_threshold=0.5, nms_threshold=0.4):
    """
    Decodes the raw output from the det_10g.onnx model.
    """
    proposals = []
    strides = [8, 16, 32]
    input_size = 640

    for i in range(3):
        stride = strides[i]
        scores = outputs[i]
        bbox_preds = outputs[i + 3]
        kps_preds = outputs[i + 6]
        
        height = input_size // stride
        width = input_size // stride
        
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchor_centers = anchor_centers.reshape(-1, 2) * stride
        
        anchor_points__x2 = np.repeat(anchor_centers[:, np.newaxis, :], 2, axis=1)
        
        scores = scores.flatten()
        bbox_preds = bbox_preds.reshape(-1, 4) * stride
        kps_preds = kps_preds.reshape(-1, 10) * stride

        keep_indices = np.where(scores > score_threshold)[0]
        if len(keep_indices) == 0:
            continue

        anchor_points_filtered = anchor_points__x2[keep_indices // 2]
        scores_filtered = scores[keep_indices]
        bbox_preds_filtered = bbox_preds[keep_indices]
        kps_preds_filtered = kps_preds[keep_indices]
        
        x1 = anchor_points_filtered[:, 0, 0] - bbox_preds_filtered[:, 0]
        y1 = anchor_points_filtered[:, 0, 1] - bbox_preds_filtered[:, 1]
        x2 = anchor_points_filtered[:, 0, 0] + bbox_preds_filtered[:, 2]
        y2 = anchor_points_filtered[:, 0, 1] + bbox_preds_filtered[:, 3]
        bboxes = np.stack([x1, y1, x2, y2], axis=1)

        kps = kps_preds_filtered + np.tile(anchor_points_filtered[:, 0, :], 5)
        
        for idx in range(len(bboxes)):
            proposals.append({
                'bbox': bboxes[idx] / r_scale,
                'landmarks': kps[idx].reshape(5, 2) / r_scale,
                'score': scores_filtered[idx]
            })

    if not proposals: return []

    proposals.sort(key=lambda x: x['score'], reverse=True)
    bboxes = np.array([p['bbox'] for p in proposals])
    scores = np.array([p['score'] for p in proposals])
    indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), score_threshold, nms_threshold)
    
    return [proposals[i] for i in indices]


def main():
    print(f"\n{C_BLUE}--- VirtuaCast Pipeline Validator ---{C_RESET}")
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputting debug images to: {os.path.abspath(output_dir)}\n")

    try:
        print("Loading ONNX models...")
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        det_session = ort.InferenceSession(os.path.join(CONFIG['model_dir'], "det_10g.onnx"), providers=providers)
        rec_session = ort.InferenceSession(os.path.join(CONFIG['model_dir'], "w600k_r50.onnx"), providers=providers)
        swap_session = ort.InferenceSession(os.path.join(CONFIG['model_dir'], "inswapper_128.onnx"), providers=providers)
        print(f"{C_GREEN}All models loaded successfully using: {ort.get_device()}{C_RESET}\n")
    except Exception as e:
        print(f"{C_RED}Error loading models: {e}{C_RESET}")
        return

    source_img = cv2.imread(CONFIG['source_image'])
    target_img = cv2.imread(CONFIG['target_image'])
    if source_img is None or target_img is None:
        print(f"{C_RED}Error: Could not load source or target image. Check paths in CONFIG.{C_RESET}")
        return

    print("1. Generating source face embedding...")
    blob, r_scale = preprocess_detection_image(source_img)
    outputs = det_session.run(None, {'input.1': blob})
    source_faces = postprocess_detection(outputs, r_scale)

    if not source_faces:
        print(f"{C_RED}Error: No face found in source image '{CONFIG['source_image']}'.{C_RESET}")
        return
    
    source_face = source_faces[0]
    
    arcface_dst_112 = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
    M_align_src = estimate_similarity_transform(source_face['landmarks'], arcface_dst_112)
    aligned_source_face = cv2.warpAffine(source_img, M_align_src, (112, 112), borderValue=0.0)
    cv2.imwrite(os.path.join(output_dir, "PY_DEBUG_01_Aligned_Source.jpg"), aligned_source_face)
    print("   - Saved: PY_DEBUG_01_Aligned_Source.jpg")
    
    rec_blob = ((aligned_source_face.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[np.newaxis, :, :, :]
    source_embedding = rec_session.run(None, {'input.1': rec_blob})[0]
    source_embedding /= np.linalg.norm(source_embedding)
    print(f"{C_GREEN}   Source embedding generated successfully.{C_RESET}\n")

    print("2. Detecting face in target frame...")
    blob, r_scale = preprocess_detection_image(target_img)
    outputs = det_session.run(None, {'input.1': blob})
    target_faces = postprocess_detection(outputs, r_scale)

    if not target_faces:
        print(f"{C_RED}Error: No face found in target image '{CONFIG['target_image']}'.{C_RESET}")
        return

    target_face = target_faces[0]
    target_img_with_box = target_img.copy()
    x1, y1, x2, y2 = [int(v) for v in target_face['bbox']]
    cv2.rectangle(target_img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for i in range(5):
        pt = tuple(target_face['landmarks'][i].astype(int))
        cv2.circle(target_img_with_box, pt, 2, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(output_dir, "PY_DEBUG_02_Target_With_Detection.jpg"), target_img_with_box)
    print("   - Saved: PY_DEBUG_02_Target_With_Detection.jpg")
    print(f"{C_GREEN}   Target face detected successfully.{C_RESET}\n")

    print("3. Aligning target and running swap model...")
    arcface_dst_128 = arcface_dst_112 * (128.0/112.0)
    M_align_tgt = estimate_similarity_transform(target_face['landmarks'], arcface_dst_128)
    aligned_target_face = cv2.warpAffine(target_img, M_align_tgt, (128, 128), borderValue=0.0)
    cv2.imwrite(os.path.join(output_dir, "PY_DEBUG_03_Aligned_Target.jpg"), aligned_target_face)
    print("   - Saved: PY_DEBUG_03_Aligned_Target.jpg")
    
    swap_blob = (aligned_target_face.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, :, :, :]
    swapped_face_raw = swap_session.run(None, {'target': swap_blob, 'source': source_embedding})[0][0]
    
    swapped_face_img = (swapped_face_raw.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    swapped_face_img = cv2.cvtColor(swapped_face_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, "PY_DEBUG_04_Generated_Swap_Face.jpg"), swapped_face_img)
    print("   - Saved: PY_DEBUG_04_Generated_Swap_Face.jpg")
    print(f"{C_GREEN}   Swap model executed successfully.{C_RESET}\n")
    
    # --- PIPELINE STEP 4: Paste Back Result (NOW WITH CORRECT ERROR HANDLING) ---
    print("4. Pasting swapped face back into target frame...")
    try:
        M_paste = cv2.invertAffineTransform(M_align_tgt)
    except cv2.error:
        print(f"\n{C_RED}CRITICAL FAILURE: The alignment matrix for the target face is non-invertible.{C_RESET}")
        print(f"{C_YELLOW}This almost always means the landmarks detected in '{CONFIG['target_image']}' are degenerate (e.g., all on one line).")
        print(f"Please try a different, clearer target image.{C_RESET}\n")
        return

    full_frame_swapped_face = cv2.warpAffine(swapped_face_img, M_paste, (target_img.shape[1], target_img.shape[0]))
    
    mask = np.full((128, 128), 255, dtype=np.uint8)
    full_frame_mask = cv2.warpAffine(mask, M_paste, (target_img.shape[1], target_img.shape[0]))
    
    full_frame_mask_blurred = cv2.GaussianBlur(full_frame_mask, (15, 15), 0)
    
    mask_float = full_frame_mask_blurred.astype(np.float32) / 255.0
    mask_float = mask_float[:, :, np.newaxis]
    
    final_result = (full_frame_swapped_face * mask_float + target_img * (1 - mask_float)).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "PY_DEBUG_08_Final_Result.jpg"), final_result)
    print("   - Saved: PY_DEBUG_08_Final_Result.jpg")
    print(f"{C_GREEN}   Blending complete.{C_RESET}\n")

    print(f"{C_GREEN}--- VALIDATION SCRIPT FINISHED ---{C_RESET}")
    print("Compare the images in the 'PY_DEBUG' folder with the ones generated by your C++ app's debug trace.")


if __name__ == '__main__':
    main()