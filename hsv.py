import cv2
import time
import argparse

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
                help="path to the (optional) image file")
args = vars(ap.parse_args())

# Global coordinates for mouse interaction
x_coord = 0
y_coord = 0

def on_mouse(event, x, y, flag, param):
    """
    Callback function for mouse events.
    Updates global coordinates on mouse move.
    """
    global x_coord
    global y_coord
    if event == cv2.EVENT_MOUSEMOVE:
        x_coord = x
        y_coord = y

cv2.namedWindow('camera')

# Main loop
while True:
    # Read image from arguments
    src = cv2.imread(args["image"])
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    
    # Set mouse callback
    cv2.setMouseCallback("camera", on_mouse)
    
    # Get HSV values at the current mouse position
    s = hsv[y_coord, x_coord]
    print("H:", s[0], "      S:", s[1], "       V:", s[2])
    
    # Display the image
    cv2.imshow("camera", hsv)
    
    # Exit on ESC key
    if cv2.waitKey(10) == 27:
        break