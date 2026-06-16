import argparse
import subprocess
import time
import os

import cv2
import insightface
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Tối ưu CPU: dùng tất cả cores
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())


def get_face(app, img_path, min_size=80):
    """Detect face in image with size check."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    
    # Try multiple det_sizes for better detection
    faces = app.get(img)
    
    if not faces:
        raise ValueError(f"No face detected in: {img_path}")
    
    # Sort by detection score (confidence) first, then by x position
    faces_sorted = sorted(faces, key=lambda f: (-f.det_score, f.bbox[0]))
    best_face = faces_sorted[0]
    
    face_w = best_face.bbox[2] - best_face.bbox[0]
    face_h = best_face.bbox[3] - best_face.bbox[1]
    
    if face_w < min_size or face_h < min_size:
        print(f"  WARNING: Face in {img_path} is small ({face_w:.0f}x{face_h:.0f}px). "
              f"Recommend using larger image (>{min_size}px).")
    
    print(f"  {img_path.name}: face at ({best_face.bbox[0]:.0f}, {best_face.bbox[1]:.0f}), "
          f"size={face_w:.0f}x{face_h:.0f}, score={best_face.det_score:.3f}")
    
    return best_face, img


def detect_faces_video(app, frame, det_score_threshold=0.5):
    """Detect faces in video frame with filtering."""
    faces = app.get(frame)
    
    # Filter by detection score
    good_faces = [f for f in faces if f.det_score >= det_score_threshold]
    
    if len(good_faces) < len(faces):
        print(f"    Filtered out {len(faces) - len(good_faces)} low-confidence faces")
    
    # Sort by x position (left to right)
    return sorted(good_faces, key=lambda f: f.bbox[0])


def upscale_face_region(frame, bbox, target_size=128):
    """Upscale face region for better swap quality."""
    x1, y1, x2, y2 = map(int, bbox)
    face_w = x2 - x1
    face_h = y2 - y1
    
    if face_w >= target_size and face_h >= target_size:
        return frame, bbox  # Already big enough
    
    # Calculate scale factor
    scale = max(target_size / face_w, target_size / face_h)
    
    # Expand ROI to include more context
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    new_w, new_h = int(face_w * scale), int(face_h * scale)
    
    x1_new = max(0, cx - new_w // 2)
    y1_new = max(0, cy - new_h // 2)
    x2_new = min(frame.shape[1], x1_new + new_w)
    y2_new = min(frame.shape[0], y1_new + new_h)
    
    # Extract and upscale
    roi = frame[y1_new:y2_new, x1_new:x2_new]
    upscaled = cv2.resize(roi, (new_w * 2, new_h * 2), interpolation=cv2.INTER_LANCZOS4)
    
    return upscaled, (x1_new, y1_new, x2_new, y2_new)


def main():
    parser = argparse.ArgumentParser(description="Deepfake: swap 3 faces into a video")
    parser.add_argument("--left", default=str(BASE_DIR / "anh" / "left.jpg"), help="Left face image (or single face if --single)")
    parser.add_argument("--mid", default=str(BASE_DIR / "anh" / "mid.jpg"), help="Middle face image")
    parser.add_argument("--right", default=str(BASE_DIR / "anh" / "right.jpg"), help="Right face image")
    parser.add_argument("--video", default=str(BASE_DIR / "ComeMyWayVideo.mp4"), help="Input video")
    parser.add_argument("--output", default=str(BASE_DIR / "output_deepfake.mp4"), help="Output video")
    parser.add_argument("--single", action="store_true", help="Single face mode: only swap 1 face (use --left for source face)")
    parser.add_argument("--target-index", type=int, default=0, help="Which face in video to swap (0=first, 1=second, etc. Only used with --single)")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame (default: 1 = all frames)")
    parser.add_argument("--det-size", type=int, default=640, help="Detection size (320=faster, 640=better)")
    parser.add_argument("--model", default="inswapper_128.onnx", help="Face swap model file")
    parser.add_argument("--det-score", type=float, default=0.5, help="Minimum face detection confidence (0.0-1.0)")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA GPU (requires onnxruntime-gpu)")
    parser.add_argument("--enhance", action="store_true", help="Apply face enhancement (CodeFormer/GFPGAN)")
    parser.add_argument("--threads", type=int, default=0, help="Number of CPU threads (0=auto=all cores)")
    args = parser.parse_args()

    # Set CPU threads
    if args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        cv2.setNumThreads(args.threads)
    else:
        cv2.setNumThreads(os.cpu_count())

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu else ["CPUExecutionProvider"]

    print(f"[INFO] CPU cores: {os.cpu_count()}")
    print(f"[INFO] Threads: {args.threads if args.threads > 0 else os.cpu_count()}")
    print(f"[INFO] Providers: {providers}")

    print("[1/5] Loading face analysis model (buffalo_l)...")
    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0 if args.gpu else -1, det_size=(args.det_size, args.det_size))

    print("[2/5] Loading face swap model...")
    # Support both old and new model naming
    model_path = Path.home() / ".insightface" / "models" / args.model
    if not model_path.exists():
        # Try without .onnx extension
        model_path = Path.home() / ".insightface" / "models" / (args.model + ".onnx")
    if not model_path.exists():
        print(f"  ERROR: Model not found at {model_path}")
        print(f"  Searched in: {Path.home() / '.insightface' / 'models'}")
        print("  Available models:")
        models_dir = Path.home() / ".insightface" / "models"
        if models_dir.exists():
            for f in models_dir.iterdir():
                if f.suffix == ".onnx":
                    print(f"    - {f.name}")
        print("\n  Download models:")
        print("  - inswapper_128: https://huggingface.co/ezioruan/inswapper_128.onnx")
        print("  - inswapper-512-live: https://github.com/deepinsight/inswapper-512-live")
        return
    
    print(f"  Using model: {model_path.name}")
    swapper = insightface.model_zoo.get_model(str(model_path), providers=providers)

    print("[3/5] Detecting faces in source images...")
    
    if args.single:
        # Single face mode: only need 1 source face
        source_face, _ = get_face(app, Path(args.left))
        source_faces = [source_face]
        print(f"  Single face mode: using {args.left}")
    else:
        # Multi-face mode: need 3 source faces
        left_face, _ = get_face(app, Path(args.left))
        mid_face, _ = get_face(app, Path(args.mid))
        right_face, _ = get_face(app, Path(args.right))
        source_faces = [left_face, mid_face, right_face]
        
        # Check if source face sizes are reasonable
        for i, name in enumerate(["left", "mid", "right"]):
            f = source_faces[i]
            fw = f.bbox[2] - f.bbox[0]
            fh = f.bbox[3] - f.bbox[1]
            if fw < 80 or fh < 80:
                print(f"  WARNING: Source face '{name}' is small ({fw:.0f}x{fh:.0f}px). "
                      f"Swap quality may be poor.")

    print("[4/5] Processing video frames...")
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"  Skip: every {args.skip} frame(s), det_size={args.det_size}")
    print(f"  Face confidence threshold: {args.det_score}")
    print(f"  Mode: {'SINGLE face' if args.single else 'MULTI face (3 faces)'}")
    if args.single:
        print(f"  Target face index: {args.target_index}")
    print(f"  Enhancement: {'ON' if args.enhance else 'OFF'}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    processed = 0
    start_time = time.time()
    last_frame = None
    
    # Track face statistics
    face_size_log = []
    swap_count = 0
    skip_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.skip == 0:
            faces = detect_faces_video(app, frame, args.det_score)
            
            if args.single:
                # Single face mode: only swap 1 face
                if len(faces) > 0:
                    target_idx = min(args.target_index, len(faces) - 1)
                    target_face = faces[target_idx]
                    
                    fw = target_face.bbox[2] - target_face.bbox[0]
                    fh = target_face.bbox[3] - target_face.bbox[1]
                    face_size_log.append((fw, fh))
                    
                    print(f"  Frame {frame_idx}: {len(faces)} faces, swapping face[{target_idx}] "
                          f"size={fw:.0f}x{fh:.0f} score={target_face.det_score:.3f}")
                    
                    if fw < 50 or fh < 50:
                        print(f"    WARNING: Face is very small ({fw:.0f}x{fh:.0f}px). Quality will be poor!")
                    
                    swapped = swapper.get(frame.copy(), target_face, source_faces[0], paste_back=True)
                    x1, y1, x2, y2 = target_face.bbox.astype(int)
                    h, w = y2 - y1, x2 - x1
                    pad_x, pad_y = int(w * 0.4), int(h * 0.4)
                    x1p = max(0, x1 - pad_x)
                    y1p = max(0, y1 - pad_y)
                    x2p = min(frame.shape[1], x2 + pad_x)
                    y2p = min(frame.shape[0], y2 + pad_y)
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    axes = (int(w * 0.6), int(h * 0.6))
                    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
                    mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
                    region = frame[y1p:y2p, x1p:x2p].astype(np.float32)
                    swap_region = swapped[y1p:y2p, x1p:x2p].astype(np.float32)
                    mask_region = mask_3ch[y1p:y2p, x1p:x2p]
                    blended = (region * (1 - mask_region) + swap_region * mask_region).astype(np.uint8)
                    frame[y1p:y2p, x1p:x2p] = blended
                    swap_count += 1
                else:
                    print(f"  Frame {frame_idx}: No faces detected! Skipping.")
                    skip_count += 1
            else:
                # Multi-face mode: swap 3 faces
                if len(faces) >= 3:
                    print(f"  Frame {frame_idx}: {len(faces)} faces detected")
                    for i, f in enumerate(faces[:3]):
                        fw = f.bbox[2] - f.bbox[0]
                        fh = f.bbox[3] - f.bbox[1]
                        face_size_log.append((fw, fh))
                        print(f"    Face {i}: size={fw:.0f}x{fh:.0f}, score={f.det_score:.3f}")
                    
                    result = frame.copy()
                    for i in range(3):
                        face_w = faces[i].bbox[2] - faces[i].bbox[0]
                        face_h = faces[i].bbox[3] - faces[i].bbox[1]
                        
                        if face_w < 50 or face_h < 50:
                            print(f"    WARNING: Face {i} is very small ({face_w:.0f}x{face_h:.0f}px).")
                        
                        swapped = swapper.get(frame.copy(), faces[i], source_faces[i], paste_back=True)
                        x1, y1, x2, y2 = faces[i].bbox.astype(int)
                        h, w = y2 - y1, x2 - x1
                        pad_x, pad_y = int(w * 0.4), int(h * 0.4)
                        x1p = max(0, x1 - pad_x)
                        y1p = max(0, y1 - pad_y)
                        x2p = min(frame.shape[1], x2 + pad_x)
                        y2p = min(frame.shape[0], y2 + pad_y)
                        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        axes = (int(w * 0.6), int(h * 0.6))
                        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
                        mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
                        region = result[y1p:y2p, x1p:x2p].astype(np.float32)
                        swap_region = swapped[y1p:y2p, x1p:x2p].astype(np.float32)
                        mask_region = mask_3ch[y1p:y2p, x1p:x2p]
                        blended = (region * (1 - mask_region) + swap_region * mask_region).astype(np.uint8)
                        result[y1p:y2p, x1p:x2p] = blended
                    frame = result
                    swap_count += 1
                else:
                    print(f"  Frame {frame_idx}: Only {len(faces)} faces found (need 3). Skipping.")
                    skip_count += 1
            
            last_frame = frame
            processed += 1
        else:
            if last_frame is not None:
                frame = last_frame

        out.write(frame)
        frame_idx += 1

        if frame_idx % 30 == 0:
            elapsed = time.time() - start_time
            speed = processed / elapsed if elapsed > 0 else 0
            remaining = (total_frames // args.skip - processed) / speed if speed > 0 else 0
            pct = frame_idx / total_frames * 100
            print(f"  Progress: {frame_idx}/{total_frames} ({pct:.1f}%) | "
                  f"{speed:.1f} frames/s | ETA: {remaining:.0f}s")

    cap.release()
    out.release()
    
    # Print statistics
    print(f"\n  Processing complete:")
    print(f"    Frames swapped: {swap_count}")
    print(f"    Frames skipped: {skip_count}")
    if face_size_log:
        avg_w = sum(s[0] for s in face_size_log) / len(face_size_log)
        avg_h = sum(s[1] for s in face_size_log) / len(face_size_log)
        print(f"    Average face size: {avg_w:.0f}x{avg_h:.0f}px")
        if avg_w < 50 or avg_h < 50:
            print(f"    WARNING: Face size is very small. Use video with larger faces for better quality.")

    elapsed = time.time() - start_time
    print(f"  Video done in {elapsed:.1f}s ({processed/elapsed:.1f} frames/s)")

    print("[5/5] Merging audio from original video...")
    tmp_video = args.output + ".tmp.mp4"
    Path(args.output).rename(tmp_video)
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", tmp_video,
            "-i", args.video,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            args.output
        ], check=True, capture_output=True)
        Path(tmp_video).unlink()
        print(f"  Output with audio: {args.output}")
    except subprocess.CalledProcessError as e:
        Path(tmp_video).rename(args.output)
        print(f"  Warning: Could not merge audio ({e.stderr.decode()})")
        print(f"  Output (video only): {args.output}")


if __name__ == "__main__":
    main()
