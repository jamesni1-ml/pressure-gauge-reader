# Analog Pressure Gauge Reader — Raspberry Pi 4 Setup Guide

## Hardware Requirements

- **Raspberry Pi 4** (8GB RAM)
- **Raspberry Pi Camera Module 2** (optional — for live capture)
- MicroSD card (32GB+ recommended)
- Power supply (USB-C, 5V 3A)

## Software Setup

### 1. Flash Raspberry Pi OS

Flash **Raspberry Pi OS 64-bit (Bookworm)** using Raspberry Pi Imager:
- Select "Raspberry Pi OS (64-bit)" — the full desktop version
- Enable SSH in imager settings if you want remote access

### 2. Enable Camera (if using Pi Camera 2)

```bash
sudo raspi-config
# Interface Options → Camera → Enable
# Reboot when prompted
```

### 3. Install System Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-opencv libatlas-base-dev libhdf5-dev
```

### 4. Install Python Dependencies

```bash
cd ~/gauge_reader/
pip install -r requirements_pi.txt
```

> **Note:** EasyOCR's first run will download language models (~100MB). Ensure you have internet access for the first run.

### 5. Verify Installation

```bash
python3 -c "import cv2; import easyocr; print('All dependencies OK')"
```

## Deployment

### Copy Files to Pi

From your training machine, copy the `rpi_deploy/` folder to the Pi:

```bash
# Using scp (from training machine):
scp -r rpi_deploy/ pi@<PI_IP>:~/gauge_reader/

# Or using USB drive, copy the rpi_deploy/ folder contents
```

The `gauge_reader/` folder on the Pi should contain:
```
~/gauge_reader/
├── gauge_reader.py        # Main inference script
├── gauge_detector.onnx    # Trained YOLOv8n model
├── requirements_pi.txt    # Python dependencies
└── README_PI.md           # This file
```

## Usage

### Batch Mode (primary use case)

Process a folder of 400 photos:

```bash
# Basic batch processing
python3 gauge_reader.py --input ./photos/ --output results.csv

# With annotated images saved
python3 gauge_reader.py --input ./photos/ --output results.csv --save-annotated

# Custom confidence threshold
python3 gauge_reader.py --input ./photos/ --output results.csv --conf 0.4
```

### Single Image Mode

```bash
python3 gauge_reader.py --single photo.jpg
python3 gauge_reader.py --single photo.jpg --save-annotated
```

### Output

**CSV file** with columns:
| Column | Description |
|--------|-------------|
| `filename` | Image filename |
| `reading` | Gauge pressure reading (numeric) |
| `scale_min` | Detected scale minimum value |
| `scale_max` | Detected scale maximum value |
| `needle_angle` | Detected needle angle (degrees) |
| `confidence` | YOLOv8 detection confidence |
| `num_markings` | Number of scale markings detected by OCR |
| `status` | OK, NO_GAUGE_DETECTED, NEEDLE_NOT_FOUND, etc. |
| `time_sec` | Processing time per image |

**Annotated images** (if `--save-annotated`): saved in `annotated/` subfolder next to the CSV.

## Performance

- **Target**: 400 images in under 1 hour
- **Budget per image**: ~9 seconds
- **Typical breakdown**:
  - YOLOv8n ONNX detection: ~1-2 sec
  - Perspective correction: ~0.3 sec
  - EasyOCR scale reading: ~2-4 sec
  - Needle detection: ~0.5 sec
  - **Total: ~4-7 sec/image**

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `easyocr` import error | `pip install easyocr` |
| Model not found | Ensure `gauge_detector.onnx` is in the same directory as the script |
| Low accuracy | Try `--conf 0.3` for more detections; ensure photos are well-lit |
| Out of memory | Close other applications; Pi 4 8GB should handle this fine |
| Camera not detected | Run `sudo raspi-config` → Interface → Camera → Enable; reboot |
| Slow processing | Normal on Pi 4 CPU — EasyOCR is the bottleneck (~2-4 sec/image) |

## Optional: Live Capture with Pi Camera 2

To capture photos for batch processing using the Pi Camera:

```bash
# Capture a single photo
libcamera-still -o photo.jpg

# Capture multiple photos at intervals (every 30 seconds, 400 photos)
for i in $(seq 1 400); do
    libcamera-still -o "photos/gauge_$(printf '%04d' $i).jpg" --timeout 1000
    sleep 30
done
```

Then process with:
```bash
python3 gauge_reader.py --input ./photos/ --output results.csv
```
