import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    """
    converts the image into grayscale
    applies gaussian blur for noise filtering
    canny edge detection for detecting intensity 
    changes between adjacent pels
    """

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    img_canny = cv2.Canny(img_blur, 50, 150)
    return img_canny

def roi(image):
    """
    return the region of interest
    for the given input image
    """
    height, width = image.shape
    region = np.array([[[200, height], [1100, height], [550, 250]]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, region, 255)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image


def make_coords(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit, right_fit = [], []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coords(image, left_fit_average)
    right_line = make_coords(image, right_fit_average)
    return np.array([left_line, right_line])


cap = cv2.VideoCapture("./data/test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    img_canny = canny(frame)
    # region of interest
    roi_img = roi(img_canny)
    # hough transform
    lines = cv2.HoughLinesP(roi_img, 2, np.pi/180, 100, np.array([]),
        minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

