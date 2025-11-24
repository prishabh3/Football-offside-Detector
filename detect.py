# USAGE
# python detect.py --input path/to/file

# Import the necessary packages
from scipy.optimize import fsolve
import numpy as np
import argparse
import imutils
import cv2
import os

# Construct the argument parse and parse the arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-i", "--input", required=True, help="path to the input file (image or video)")
arg_parser.add_argument("-d", "--debug", action="store_true", help="enable debug mode to visualize detected lines")
args = vars(arg_parser.parse_args())

# Define shirt colors of the teams and the pitch, hsv color model
# team1 = white, team2 = red
lower_hsv_ranges = {'team1': (0, 0, 180), 'green': (35, 50, 50), 'team2': (120, 40, 85)}
upper_hsv_ranges = {'team1': (255, 40, 255), 'green': (60, 255, 255), 'team2': (186, 255, 255)}

# Thresholds for line detection
LOW_GRAY_THRESHOLD = 50
HIGH_GRAY_THRESHOLD = 150

# Initialize morph-kernel
kernel = np.ones((5, 15), np.uint8)


def f(xy, funcargs):
    """
    Function for intersection calculation.
    """
    x, y = xy
    z = np.array([y - funcargs[0][0] * x - funcargs[0][1], y - funcargs[0][2] * x - funcargs[0][3]])
    return z


def color_filtering(image, lower_threshold, upper_threshold):
    """
    Apply color filtering and morphological operations to detect specific colors.
    Returns the percentage of the detected color in the image.
    """
    hsv_team = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_team = cv2.inRange(hsv_team, lower_threshold, upper_threshold)
    mask_team = cv2.morphologyEx(mask_team, cv2.MORPH_OPEN, kernel)
    mask_team = cv2.morphologyEx(mask_team, cv2.MORPH_CLOSE, kernel)
    perc_team = (mask_team > 254).sum() / (len(mask_team) * len(mask_team[0]))
    return perc_team


def process_frame(frame, hog):
    """
    Process a single frame (image) to detect players and offside lines.
    """
    # Load the image and resize it to improve detection accuracy
    # Since the most of our source material is 720p and players are about
    # 1/5th of the picture we need to scale it larger to get to the 96px height.
    # Copy image for later drawings
    t = int(frame.shape[1] * 1.4)
    image = imutils.resize(frame, width=t)
    orig = image.copy()

    # Line detection: gray image / fill lines / canny / HoughLinesP
    gray_img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    blurred_gray_img = cv2.GaussianBlur(gray_img, (15, 15), 0)
    canny_edges = cv2.Canny(blurred_gray_img, LOW_GRAY_THRESHOLD, HIGH_GRAY_THRESHOLD)
    canny_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel)

    # Creating a blank to draw lines on
    line_image = np.copy(image) * 0

    # Parameters for HoughLinesP function
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 90  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 200  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(canny_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    box16_line = 0
    
    # Blank arrays for off-side lane calculation
    offside_line = np.array([])
    box5_line = np.array([])

    if args["debug"]:
        print(f"Debug: HoughLinesP detected {len(lines) if lines is not None else 0} lines.")

    # Loop for initializing the offside line.
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Debug: Draw all detected lines in blue
                if args["debug"]:
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # we only need one line. but it has be a vertical line.
                # This logic finds the longest vertical line for the "offside line" candidates
                if (x2 - x1) + 10 < (y2 - y1):
                    if box16_line < np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2):
                        box16_line = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        offside_line = (x1, y1, x2, y2)
                
                # Logic for box5 line (reference line on the right)
                # Original strict check: if canny_edges.shape[1] - x2 < 20:
                # New relaxed check: Find the right-most line in the right half of the image
                if x2 > canny_edges.shape[1] * 0.5:
                    if len(box5_line) == 0 or x2 > box5_line[2]:
                        box5_line = (x1, y1, x2, y2)

    if args["debug"]:
        if len(offside_line) > 0:
            print("Debug: Found potential offside line (vertical).")
        else:
            print("Debug: No valid offside line found.")
        
        if len(box5_line) > 0:
            print("Debug: Found potential box5 line (near right edge).")
        else:
            print("Debug: No valid box5 line found (needs to be within 20px of right edge).")

    # Calculate line parameters of the Offside / Origin Line from the two dots given from the HoughLinesP
    intersect = None
    if offside_line is not None and len(offside_line) > 0:
        line_x = (offside_line[3] - offside_line[1]) / (offside_line[2] - offside_line[0])
        line_b = offside_line[1] - line_x * offside_line[0]
        t1y = 0
        t1x = int(line_b)
        t2y = int(image.shape[1])
        t2x = int(line_x * image.shape[1] + line_b)

        # Calculate intersection of 16m box and 5m box. 
        # Calculated point is used for turning offsideLine into right angle
        if box5_line is not None and len(box5_line) > 0:
            test_x = (box5_line[3] - box5_line[1]) / (box5_line[2] - box5_line[0])
            test_b = box5_line[1] - test_x * box5_line[0]
            z = (line_x, line_b, test_x, test_b)

            intersect = fsolve(f, [1, 2], args=[z])

    # Init arrays to store the players after it was made sure that it is indeed a player
    team1_player = np.array([])
    team2_player = np.array([])

    # Store the position of the last man of each team on the pitch
    team1_player_position_x = 0
    team2_player_position_x = 100000000
    team1_player_position_y = 0
    team2_player_position_y = 0

    # Detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4))

    # Iterate over the found players
    for i in range(len(weights)):

        (x, y, w, h) = rects[i]

        # DEBUG: Show all detected pedestrians
        # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Throw all results under a threshold away
        if weights[i] > 0.7:

            # Crop an rectangle of the result
            crop = orig[y:y + h, x:x + w]

            # Find the percentage of the definded colors
            # Team 1
            perc_team1 = color_filtering(crop, lower_hsv_ranges['team1'], upper_hsv_ranges['team1'])

            # DEBUG:
            # print("Team1/white " + str((maskTeam1>254).sum() / (len(maskTeam1)*len(maskTeam1[0]))))

            # Grass
            perc_green = color_filtering(crop, lower_hsv_ranges['green'], upper_hsv_ranges['green'])

            # DEBUG:
            # print("Green " + str((maskGreen>254).sum() / (len(maskGreen)*len(maskGreen[0]))))

            # Team2
            perc_team2 = color_filtering(crop, lower_hsv_ranges['team2'], upper_hsv_ranges['team2'])

            # DEBUG:
            # print("Team2/red " + str((maskTeam2>254).sum() / (len(maskTeam2)*len(maskTeam2[0]))))

            # DEBUG: apply a blue rectange on all found spots
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Use the green to filter out fans
            if perc_green < 0.7 and perc_green > 0.1:
                # Sort players into the teams
                if perc_team1 > 0.03:
                    team2_player = np.append(team2_player, rects[i])
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    # Calculate offside line from found player
                    if box5_line is not None and len(box5_line) > 0 and intersect is not None:
                        line_x = (intersect[1] - (y + h)) / (intersect[0] - (w + x))
                        line_b = (y + h) - line_x * (x + w)
                        ro1y = 0
                        ro1x = int(line_b)
                        ro2y = int(image.shape[1])
                        ro2x = int(line_x * image.shape[1] + line_b)

                        if team2_player_position_x > ro2x and ro2x > 0:
                            team2_player_position_x = ro2x
                            team2_player_position_y = ro1x
                else:
                    team1_player = np.append(team1_player, rects[i])
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # draw the Origin if the Offside Line

    # DEBUG:
    # cv2.line(image, (t1y, t1x), (t2y,t2x), (100,240,80), 5)

    if box5_line is not None and len(box5_line) > 0:
        cv2.line(image, (0, team2_player_position_y), (int(image.shape[1]), team2_player_position_x), (120, 120, 220), 5)
    elif args["debug"]:
        print("Debug: Skipping drawing offside line because box5_line was not found.")

    # DEBUG:
    # draw line on the 5 box
    # if (len(box5Line) > 0):
    # 	cv2.line(image, (box5Line[0], box5Line[1]), (box5Line[2],box5Line[3]), (30,120,120), 5)

    # Show the result
    image = imutils.resize(image, height=720)
    cv2.imshow("Players and Offside", image)
    return image


def main():
    # Initialize the HOG descriptor/person detector
    # Interesting that most variables only support one parameter right now
    win_size = (48, 96)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    # Use Daimler detector because it is trained to recognise people 48x96 pixels
    hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())

    input_path = args["input"]
    
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    # Try to open as image first
    image = cv2.imread(input_path)
    
    if image is not None:
        print(f"Processing '{input_path}' as an image...")
        process_frame(image, hog)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # If not an image, try to open as video
    vid_cap = cv2.VideoCapture(input_path)
    
    if vid_cap.isOpened():
        print(f"Processing '{input_path}' as a video...")
        # Loop over the video, processing frames
        while True:
            # Read frame from video
            (grabbed, frame) = vid_cap.read()
            
            # If we are viewing a video and we did not grab a frame,
            # then we have reached the end of the video
            if not grabbed:
                break

            process_frame(frame, hog)

            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                break

        # Close Video
        vid_cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"Error: Could not open '{input_path}' as image or video.")


if __name__ == '__main__':
    main()




