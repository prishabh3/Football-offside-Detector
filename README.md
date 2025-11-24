# Football Offside Detection System

This project implements an automated offside line detection system using **Python**, **NumPy**, and **OpenCV**.

## Reference
The algorithm is inspired by the research paper:
[Cheshire, Halasz, Perin - Stanford Project Report](https://web.stanford.edu/class/ee368/Project_Spring_1415/Reports/Cheshire_Halasz_Perin.pdf)

## Prerequisites
Ensure you have the following installed:
* **Python 3.6.5**
* **OpenCV 3.4.1**

### Required Packages
Install dependencies via pip:
`scipy`, `numpy`, `argparse`, `opencv-python`

## How to Run

### Usage
To run the detection on a file (video or image):
```bash
python3 detect.py -i path/to/file
```
Examples:
```bash
python3 detect.py -i Offside_normal.mp4
python3 detect.py -i photo.jpg
```

### Debug Mode
To visualize all detected lines and see diagnostic information, add the `-d` or `--debug` flag:
```bash
python3 detect.py -i photo.jpg --debug
```

> **Note:** The script automatically detects if the input is an image or a video.

### Color Thresholding Tool
Use `hsv.py` to find the best HSV thresholds for team colors:
```bash
python3 hsv.py -i Offside_normal.mp4
```

## Limitations
* **Video Support:** Currently optimized for the provided example video.
* **Field Lines:** Requires at least two visible field lines to calculate the offside line.
* **Player Detection:** The detection accuracy may need further tuning.
