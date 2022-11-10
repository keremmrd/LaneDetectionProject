import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
            slope, intercept = 0.001, 0
    y1 = image.shape[0]
    y2 = int(y1*(4.2/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    try:
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])
    except Exception as e:
        print(e,'\n')
        return None

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel, kernel), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    interest_area = np.array([
    [(290, height), (1070, height), (780, 565), (670, 565)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, interest_area, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (1, 255, 1), 4)
    return line_image


cap = cv2.VideoCapture("highwayTest.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 40, np.array([]), minLineLength=20, maxLineGap=50)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    double_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result",double_image)



    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
