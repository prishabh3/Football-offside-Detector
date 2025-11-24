# Football Offside Detection System

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)

An automated system to detect offside lines in football match footage using computer vision techniques. This project processes both images and videos to identify players, field lines, and the virtual offside line.

## 🚀 Features

- **Dual Media Support**: Seamlessly handles both **Images** (`.jpg`, `.png`) and **Videos** (`.mp4`).
- **Automatic Mode Detection**: Automatically detects the input type and processes it accordingly.
- **Player Detection**: Uses HOG (Histogram of Oriented Gradients) + SVM to detect players on the field.
- **Team Segmentation**: Distinguishes between teams and the pitch using HSV color thresholding.
- **Virtual Offside Line**: Calculates and draws the offside line based on the last defender's position and vanishing point geometry.
- **Debug Mode**: Visualize internal detection logic, including Hough lines, bounding boxes, and vanishing points.
- **Calibration Tool**: Includes `hsv.py` to help you find the perfect color thresholds for different matches.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/prishabh3/Football-offside-Detector.git
    cd Football-offside-Detector
    ```

2.  **Install dependencies**:
    Ensure you have Python 3 installed. Then run:
    ```bash
    pip install -r requirements.txt
    ```
    *Dependencies include: `numpy`, `opencv-python`, `scipy`, `imutils`.*

## 💻 Usage

### Offside Detection
Run the main detection script with your input file.

**For Images:**
```bash
python3 detect.py -i assets/photo.jpg
```

**For Videos:**
```bash
python3 detect.py -i assets/match_video.mp4
```

**Debug Mode:**
Add the `--debug` (or `-d`) flag to see what's happening under the hood:
```bash
python3 detect.py -i assets/photo.jpg --debug
```

### Color Calibration
If the detection isn't working well for a specific video (e.g., different jersey colors), use the HSV tool to find new thresholds:
```bash
python3 hsv.py -i assets/match_video.mp4
```
*Click on the video window to see the HSV values of the pixel under your cursor.*

## 🧠 How It Works

1.  **Preprocessing**: The input is resized, and a Gaussian blur is applied to reduce noise.
2.  **Field Line Detection**: 
    - Converts the image to grayscale.
    - Uses **Canny Edge Detection** to find edges.
    - Applies the **Hough Transform** (`cv2.HoughLinesP`) to identify straight lines (field markings).
3.  **Vanishing Point Calculation**:
    - Filters for vertical-ish lines to find the "Offside Line" candidates.
    - Filters for horizontal-ish lines to find the "Box Line".
    - Calculates the intersection to determine the perspective and vanishing point.
4.  **Player Detection**:
    - Uses a pre-trained **HOG + SVM** detector (Daimler People Detector) to find upright human figures.
5.  **Team Classification**:
    - Analyzes the color histogram of each detected player bounding box.
    - Uses **HSV Thresholding** to classify players into Team 1, Team 2, or background (grass/fans).
6.  **Offside Line Rendering**:
    - Identifies the last defender of the defending team.
    - Draws a line through that player's position, adjusted for the calculated perspective (vanishing point).

## ⚠️ Limitations

- **Camera Angle**: Works best with broadcast-style camera angles (side view).
- **Line Visibility**: Requires at least two visible field lines (e.g., 16m box and side line) to calculate perspective.
- **Color Dependency**: Relies on distinct jersey colors. If teams wear similar colors to the pitch or each other, calibration is required.

## 📚 Reference

This algorithm is inspired by the research paper:
> **Automatic Offside Detection in Soccer Images**  
> *Cheshire, Halasz, Perin - Stanford Project Report*  
> [Read the Paper](https://web.stanford.edu/class/ee368/Project_Spring_1415/Reports/Cheshire_Halasz_Perin.pdf)
