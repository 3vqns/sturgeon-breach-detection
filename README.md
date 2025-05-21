# Sturgeon Breach Detection using YOLOv8

This project implements a custom-trained YOLOv8 object detection model to identify instances of sturgeon breaching (leaping out of the water) from video footage. Developed in collaboration with CIBiLI, this system aims to enhance sturgeon conservation efforts by automating the detection process and eliminating the need for manual video review.

## Key Features

- Accurate breach detection using a fine-tuned YOLOv8 model
- Automatic timestamp logging in HH:MM:SS format
- Built-in filtering to avoid redundant detections within a two-second window
- Cross-platform Python implementation with minimal setup requirements

## Project Structure

```
YOLOv8_Project/
├── BreachDetection.py         # Main detection script
├── requirements.txt           # Project dependencies
├── SturgeonBest.pt            # Trained YOLOv8 weights
├── README.md                  # Project documentation
└── videos/                    # Directory for input video files
```

---

## Output

Detected breaches are optionally saved as a new video file with bounding boxes drawn around identified breaches. The output video is stored in the `outputVideos/` directory as `output_with_boxes.mp4`. Timestamp data for breach events is printed to the terminal in HH:MM:SS format.

---

## Example Usage

Example run:
```bash
python BreachDetection.py
```
Sample input:
```
Insert Video Filename (e.g., SturgeonVideo.mp4): example_clip.mp4
Insert Confidence Level (0.0 - 1.0): 0.3
Do you want to save the new video with bounding boxes? (y/n): y
```

Terminal output:
```
Running breach detection...
Number of breaches detected: 5
Timestamps (h:m:s): 00:00:03, 00:00:07, 00:00:12, 00:00:15, 00:00:21
Processing time: 0h 0m 18.74s
```

---

## Dependencies

Key libraries used:
- `ultralytics`: YOLOv8 detection framework
- `opencv-python`: video processing and annotation
- `matplotlib`, `numpy`: auxiliary support

See `requirements.txt` for exact versions.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sturgeon-breach-detection.git
cd sturgeon-breach-detection
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute the Detection Script

Place the target video file in a folder (e.g., `videos/`) you create manually, and ensure the file path is correctly referenced in `BreachDetection.py`:
```bash
python BreachDetection.py
```

## Model Information

The `SturgeonBest.pt` file represents a YOLOv8 model trained on annotated imagery of sturgeon breaches, prepared and exported using Roboflow for compatibility with the Ultralytics YOLOv8 framework.

## Additional Notes

- High-quality video inputs are recommended to maximize detection accuracy.
- Output timestamps are printed to the terminal; modify the script to export these to a file if needed.

## Acknowledgments

This project was developed by Evans Armantrading III as part of ongoing research at CIBiLI to support the conservation of endangered sturgeon species through advanced AI-based monitoring systems.