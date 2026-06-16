# Deepfake Face Swap Script

Ghép khuôn mặt vào video. Hỗ trợ cả chế độ **1 khuôn mặt** (single) và **3 khuôn mặt** (multi).

## Cài đặt

```bash
# Cài dependencies bằng uv
uv sync

# Download model (chọn 1 trong 2)

## Model cũ: inswapper_128 (128×128, nhanh, nhẹ)
mkdir -p ~/.insightface/models
wget -O ~/.insightface/models/inswapper_128.onnx \
  "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"

## Model mới: inswapper-512-live (512×512, sắc nét, nhẹ hơn 10x tài nguyên)
# Tải từ: https://github.com/deepinsight/inswapper-512-live
# Hoặc tìm trên HuggingFace: "inswapper-512-live"
```

## Sử dụng

### Chế độ 1 khuôn mặt (Khuyên dùng ⭐)

Chỉ swap **1 face** trong video. Phù hợp khi video chỉ có 1 người, face lớn, rõ nét.

```bash
# Cơ bản - chỉ cần 1 ảnh source và 1 video
uv run python deepfake.py --single --left anh/face.jpg --video input.mp4

# Chọn face thứ N trong video (nếu có nhiều người)
uv run python deepfake.py --single --target-index 1 --left anh/face.jpg --video input.mp4

# Tối ưu chất lượng - xử lý full frame, det_size cao
uv run python deepfake.py --single --left anh/face.jpg --skip 1 --det-size 640

# Dùng model 512 (nếu có)
uv run python deepfake.py --single --left anh/face.jpg --model inswapper-512-live.onnx
```

**Ưu điểm single mode:**
- Face trong video **lớn hơn** (camera focus 1 người)
- Chỉ cần detect **≥1 face** (không cần đủ 3)
- **Không cần sort** left/mid/right
- **Chất lượng swap cao hơn** (không bị chia nhỏ vùng face)
- **Nhanh hơn** (chỉ swap 1 lần/frame)

### Chế độ 3 khuôn mặt (Multi-face)

Swap **3 faces** vào video có 3 người đứng cạnh nhau.

```bash
# Chạy cơ bản (xử lý tất cả frames, det_size=640)
uv run python deepfake.py

# Chạy nhanh hơn: skip frame + giảm det_size
uv run python deepfake.py --skip 3 --det-size 320

# Chỉ định file cụ thể
uv run python deepfake.py --left anh/face1.jpg --mid anh/face2.jpg --right anh/face3.jpg --video input.mp4 --output result.mp4

# Dùng model mới (nếu đã tải inswapper-512-live)
uv run python deepfake.py --model inswapper-512-live.onnx

# Dùng GPU (cần cài onnxruntime-gpu)
uv run python deepfake.py --gpu
```

## Tham số

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--left` | `anh/left.jpg` | Ảnh khuôn mặt source (trái hoặc single) |
| `--mid` | `anh/mid.jpg` | Ảnh khuôn mặt giữa |
| `--right` | `anh/right.jpg` | Ảnh khuôn mặt phải |
| `--video` | `ComeMyWayVideo.mp4` | Video đầu vào |
| `--output` | `output_deepfake.mp4` | Video đầu ra |
| `--single` | off | **Chế độ 1 khuôn mặt** ⭐ |
| `--target-index` | `0` | Chọn face thứ N trong video (chỉ với `--single`) |
| `--skip` | `1` | Xử lý mỗi N frame (3 = nhanh gấp 3) |
| `--det-size` | `640` | Kích thước detect (320=nhanh, 640=chính xác, 1280=chi tiết) |
| `--model` | `inswapper_128.onnx` | Model face swap |
| `--det-score` | `0.5` | Ngưỡng độ tin cậy detect (0.0-1.0, cao hơn = khắt khe hơn) |
| `--gpu` | off | Dùng CUDA GPU |
| `--enhance` | off | Bật face enhancement (cần setup thêm) |
| `--threads` | `0` | Số CPU threads (0 = auto) |

## Mẹo tăng tốc

- `--skip 3 --det-size 320`: Nhanh ~5-10x, chất lượng hơi giảm
- `--skip 5 --det-size 320`: Nhanh ~10-15x, chất lượng giảm nhiều hơn
- `--gpu`: Nhanh 10-50x nếu có GPU NVIDIA + onnxruntime-gpu
- `--model inswapper-512-live.onnx`: Model mới 512×512, nhẹ hơn 10x tài nguyên

## So sánh Model

| Model | Độ phân giải | Tốc độ | Chất lượng | Ghi chú |
|-------|-------------|--------|-----------|---------|
| `inswapper_128` | 128×128 | ⚡ Nhanh | ⭐⭐⭐ | Model cũ, mờ khi zoom |
| `inswapper-512-live` | 512×512 | ⚡⚡⚡ Rất nhanh | ⭐⭐⭐⭐⭐ | **Model mới 2025**, nhẹ hơn 10x |
| `inswapper_512` | 512×512 | ⚡ Nhanh | ⭐⭐⭐⭐⭐ | Production model |
| `HyperSwap 1B` | 256×256 | ⚡⚡ Nhanh | ⭐⭐⭐⭐⭐ | Của FaceFusion |

> 💡 **Khuyến nghị**: Nếu muốn video mượt + sắc nét, hãy dùng **inswapper-512-live** (512×512, ra mắt 03/2025). Output gấp 16 lần pixel so với 128×128 nhưng tài nguyên tính toán lại ít hơn 10 lần!

## 💡 Mẹo làm video mượt và đẹp hơn

### 1. Chọn video phù hợp (quan trọng nhất!)

| Yếu tố | Tốt | Kém |
|--------|-----|-----|
| **Số người** | 1 người (dùng `--single`) | 3+ người xa nhau |
| **Kích thước face** | Face chiếm 1/4 màn hình | Face nhỏ, xa camera |
| **Ánh sáng** | Đều, rõ | Tối, chói, nhiều bóng |
| **Góc quay** | Chính diện hoặc gần chính diện | Nghiêng quá 45° |
| **Chuyển động** | Ít chuyển động đầu | Liên tục quay đầu |

### 2. Chế độ 1 khuôn mặt (Single) - Khuyên dùng

```bash
# Video chỉ có 1 người, face lớn, rõ nét
uv run python deepfake.py --single --left anh/face.jpg --video closeup.mp4 --skip 1 --det-size 640
```

**Tại sao single mode tốt hơn?**
- Face trong video **lớn hơn** → model swap chính xác hơn
- Không cần phân biệt left/mid/right → **không bị nhầm** face
- Chỉ swap 1 lần/frame → **nhanh hơn**, ít lỗi hơn
- Giữ được **chi tiết** (kính, râu, nốt ruồi) tốt hơn

### 3. Tăng độ chính xác detect

```bash
# det_size cao hơn = detect chính xác hơn
uv run python deepfake.py --det-size 1280

# Ngưỡng tin cậy cao hơn = chỉ swap face rõ ràng
uv run python deepfake.py --det-score 0.7
```

### 4. Giải pháp tối ưu nhất cho chất lượng cao

```bash
# Kết hợp: single mode + model 512 + full frame + det_size cao
uv run python deepfake.py \
  --single \
  --left anh/face.jpg \
  --video closeup_1nguoi.mp4 \
  --model inswapper-512-live.onnx \
  --skip 1 \
  --det-size 640 \
  --output output_hd.mp4
```

### 5. Nếu không có GPU - tối ưu CPU

```bash
# Dùng model 512 nhưng skip 2 frame để cân bằng
uv run python deepfake.py \
  --single \
  --left anh/face.jpg \
  --model inswapper-512-live.onnx \
  --skip 2 \
  --det-size 640 \
  --threads 8
```

## 🖥️ Tình trạng GPU AMD

Bạn đang dùng **AMD Ryzen 7 H 255 + Radeon 780M** (iGPU). Đáng tiếc:

| Tính năng | Tình trạng |
|-----------|-----------|
| `onnxruntime` GPU | ❌ Chỉ hỗ trợ NVIDIA CUDA |
| ROCm (AMD GPU) | ❌ Rất khó cài trên Linux |
| DirectML | ❌ Chỉ Windows |

**Kết luận**: Đang chạy **CPU-only**, nhưng Ryzen 7 của bạn rất mạnh (8 cores, 16 threads) + 17GB RAM. Dùng `--threads 0` để tự động dùng tất cả cores.

## 📊 Ví dụ workflow theo use case

### Use case 1: Video TikTok/Reels ngắn, 1 người
```bash
uv run python deepfake.py --single --left anh/face.jpg --video tiktok.mp4 --skip 1
```

### Use case 2: Video podcast/interview, 1 người nói chuyện
```bash
uv run python deepfake.py --single --left anh/face.jpg --video podcast.mp4 --skip 2 --det-size 640
```

### Use case 3: Video 3 người đứng cạnh nhau (nhóm, band)
```bash
uv run python deepfake.py --left anh/left.jpg --mid anh/mid.jpg --right anh/right.jpg --video band.mp4 --skip 2
```

### Use case 4: Chất lượng cao nhất (chậm hơn)
```bash
uv run python deepfake.py --single --left anh/face.jpg --model inswapper-512-live.onnx --skip 1 --det-size 1280
```

## Output

Video kết quả được lưu tại `output_deepfake.mp4` (hoặc tên chỉ định qua `--output`).
Audio từ video gốc được tự động ghép vào output.
