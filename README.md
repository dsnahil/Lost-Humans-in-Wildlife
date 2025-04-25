# Solo Hunter: Drone‑Based Human Detection

**A Python toolkit for high‑recall detection of lost or camouflaged humans in wildlife imagery captured by drones**

---

## 🚀 Project Overview
Solo Hunter provides an end‑to‑end pipeline to detect people in aerial images of dense forests, rocky terrain, and other challenging environments. Leveraging YOLOv8, tiling with overlap, test‑time augmentation (TTA), and custom post‑filtering, this project maximizes recall on tiny, camouflaged targets while suppressing false alarms from rocks, stumps, and foliage. The dataset can be found here but it is very limited to be shared openly: https://drive.google.com/drive/folders/1tmPlEg5DPaduGL592t3h6VYvDB79xn1p?usp=sharing 

Key innovations:
- **Adaptive Tiling with Overlap**: 3×3 grid with 25% overlap ensures boundary objects aren’t missed.  
- **Batched High‑Resolution TTA**: Run YOLOv8 on batches of high‑res tiles (1,536 px+), augmenting each tile at inference (flips, scales).  
- **Custom NMS & Filters**: Merge detections across tiles with IoU‑based NMS and reject unlikely candidates by size & aspect ratio.
- Uses an **Albumentations pipeline to enhance images—applying techniques** such as CLAHE for contrast enhancement, sharpening, and uniform resizing (to 640×640).
- Adds **extra augmentation transforms** such as random scaling, brightness/contrast adjustments, and horizontal flips.
- Includes **routines to “clean” annotations** by checking descriptions and tags against a predefined set of keywords (e.g., "human", "walker") to standardize and ensure that only human-related annotations are used.
- Provides a function for **tiled inference where large images are subdivided into overlapping tiles** to capture small or distant humans that might otherwise be missed in a single pass.

---

## 🔧 Features

1. **Data Augmentation**  
   - Horizontal/vertical flips, 90° rotations, brightness/contrast jitter, Gaussian noise.

2. **Tiling & Overlap**  
   - Dynamically slice each image into overlapping tiles to catch edge cases.

3. **High‑Res Inference + TTA**  
   - Batch inference of tiles at up to 2 K resolution using YOLOv8’s built‑in augmentations.

4. **Custom Merging & Filtering**  
   - Merge via `torchvision.ops.nms` and post‑filter by box area & aspect ratio to remove rock/stump false positives.

5. **Visualization**  
   - Draw bounding boxes with confidence scores and save detailed outputs for review.

---

## 🛠 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-org/eagle-eyes.git
   cd eagle-eyes
   ```

2. **Create & activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate     # Windows PowerShell
   ```
  You can also ask me for access to the venv that I used here: https://drive.google.com/drive/folders/1LbvaagXmA42o-X1rlKz8_2bFNVTZlPMR?usp=drive_link 

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

> **requirements.txt** should include at least:
> ```text
> ultralytics
> torch
> opencv-python
> albumentations
> torchvision
> tqdm
> matplotlib
> ```

---

## ⚙️ Configuration

All paths and hyperparameters live at the top of the script:

- `SOURCE_FOLDER`  – raw drone imagery directory
- `AUG_FOLDER`     – (optional) augmented images output
- `OUTPUT_FOLDER`  – detection results directory

- **Tiling**: `GRID_ROWS`, `GRID_COLS`, `OVERLAP`
- **Inference**: `IMG_SIZE`, `CONF_THRESHOLD`, `IOU_THRESHOLD`, `BATCH_SIZE`
- **Filtering**: `MIN_BOX_AREA`, `MIN_ASPECT_RATIO`, `MAX_ASPECT_RATIO`

Adjust these to suit your GPU memory and desired precision/recall trade‑off.

---

## 📥 Usage

1. **Augment images**
   ```bash
   python eagle_eyes_pipeline.py --augment
   ```
2. **Run inference**
   ```bash
   python eagle_eyes_pipeline.py --infer
   ```
3. **View results**
   - Outputs saved under `OUTPUT_FOLDER`; open them in any image viewer.

> *Tip:* If you skip augmentation, ensure there are raw images in `SOURCE_FOLDER`.

---

## 🔍 How It Works

1. **Augmentation**  – Creates variations to expand your dataset.  
2. **Tiling**        – Splits each image into 3×3 overlapping patches.  
3. **Batched TTA**   – Feeds batches of tiles through YOLOv8 at high resolution with augmentation at inference.  
4. **Merge + NMS**   – Combines all tile detections and applies non‑max suppression to merge boxes.  
5. **Post‑Filter**   – Discards detections that are too small or have unlikely aspect ratios.  
6. **Visualization** – Draws final boxes + confidences and exports annotated images.

---

## 📈 Results & Evaluation

- **Recall** on tiny, camouflaged humans improves by > 20% compared to single‑shot detection.  
- **Precision** maintained by filtering out rock/stump false positives.

Some of the ouput images look like this:

![image](https://github.com/user-attachments/assets/d6040d95-8b04-45f7-8f3e-ab43082061d0)

![image](https://github.com/user-attachments/assets/8550f796-f197-49c2-923f-62b2b65924fa)

![image](https://github.com/user-attachments/assets/5a3e2497-92d4-407e-a6e6-0ce2d1e04170)

![image](https://github.com/user-attachments/assets/8dd619cc-a6ce-4f6e-86bf-b5541ab96c87)

![image](https://github.com/user-attachments/assets/0e82747b-c093-4bd9-ad54-2fc0bf0dec2e)

![image](https://github.com/user-attachments/assets/c58306c6-6c6e-496d-9f97-78860b6ac1c3)

![image](https://github.com/user-attachments/assets/88374797-1232-4b8f-a08c-bf0bf4b1cbe3)

![image](https://github.com/user-attachments/assets/dcc6b635-5163-4d57-9db7-f25d8a0109f3)

> *Benchmark on your own annotated set to find optimal `CONF_THRESHOLD` & filter parameters.*

---

## 🔮 Future Improvements

- **Custom Fine‑Tuning**: train on labelled drone‑specific data for even better accuracy.  
- **Hard‑Example Mining**: iteratively label & retrain on model’s mistakes.  
- **Asynchronous I/O**: overlaps disk reads/writes with GPU inference for throughput gains.  
- **Alternate Architectures**: explore YOLO‑NAS or two‑stage detectors for small‑object detection.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Crafted for search‑and‑rescue enthusiasts and drone developers. Let’s find those lost hikers!*
