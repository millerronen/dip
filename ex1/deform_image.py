import cv2 as cv
import sys

try:
    imageFile = sys.argv[1]
except IndexError:
    raise SystemExit(f"usage: {sys.argv[0]} <name_of_image_file>")

image = cv.imread(imageFile)
nearestImage = cv.imread(imageFile)
biLinearImage = cv.imread(imageFile)
cubicImage = cv.imread(imageFile)

# Extracting the height and width of an image
rows, cols, _ = image.shape

window_name = 'Image'
window_name1 = 'nearest'
window_name2 = 'bi-linear'
window_name3 = 'cubic'

# colors in BGR
yellowColor = (0, 255, 255)
blackColor = (0, 0, 0)

# Line thickness of 3px
line_thickness = 3

text_thickness = 1

# Font for instruction text
font = cv.FONT_HERSHEY_DUPLEX
fontScale = 1

# text origin
textOrigin = (10, 100)

firstInstructionText = "Please place a working rectangle of interest, press 'c' to cont."
secondInstructionText = "Bend the middle line to the right/left, press 'c' to continue"


# Defining the event listener (callback function)
def drag_selection(event, x, y, flags, param):
    global line_coordinates

    if event == cv.EVENT_LBUTTONDOWN:
        line_coordinates = [(x, y)]

    # Storing (x2,y2) coordinates when the left mouse button is released and make a rectangle on the selected region
    elif event == cv.EVENT_LBUTTONUP:
        line_coordinates.append((x, y))
        start_selection_point_x = line_coordinates[0][0]
        end_selection_point_x = line_coordinates[1][0]

        line_start_point_x = int(coordinates[0][0] + (coordinates[1][0] - coordinates[0][0]) / 2)
        rectangle_cols = coordinates[1][0] - coordinates[0][0]
        rectangle_rows = coordinates[1][1] - coordinates[0][1]
        half_w = rectangle_cols / 2
        d = end_selection_point_x - start_selection_point_x

        if line_start_point_x <= start_selection_point_x <= line_start_point_x + 10:
            if coordinates[1][0] > end_selection_point_x > coordinates[0][0]:

                for row in range(coordinates[0][1], coordinates[1][1]):
                    for col in range(coordinates[0][0], coordinates[1][0]):
                        dy = abs(col - line_start_point_x)
                        dx = abs(int(coordinates[0][1] + (rectangle_rows / 2) - row))
                        d_current = float(((d * dx) / half_w) - d)
                        x_original = float(col + d_current * (1 - ((int(dy) ** 2) / (int(half_w) ** 2))))

                        calc_nearest_neighbor_interpolation(row, col, x_original)
                        calc_bi_linear_interpolation(row, col, x_original)
                        calc_cubic_interpolation(row, col, x_original)

                # Show the images in the same window - horizontally
                cv.namedWindow(window_name1)
                cv.imshow(window_name1, nearestImage)
                cv.namedWindow(window_name2)
                cv.imshow(window_name2, biLinearImage)
                cv.namedWindow(window_name3)
                cv.imshow(window_name3, cubicImage)

            else:
                print("End Selection point is out of the rectangle!!!")
        else:
            print("You have missed the middle line!")


def calc_nearest_neighbor_interpolation(row, col, x_original):
    nearestImage[row][col] = image[row][int(x_original + 0.5)]


def calc_bi_linear_interpolation(row, col, x_original):
    t = float(x_original - int(x_original))

    biLinearImage[row][col] = (t * (image[row][int(x_original)])) + (
            (1 - t) * (image[row][int(x_original + 0.5)]))
    biLinearImage[row][col] = (t * (image[row][int(x_original)])) \
                              + ((1 - t) * (image[row][int(x_original + 0.5)]))


def calc_cubic_interpolation(row, col, x_original):
    t = float(x_original - int(x_original))
    a1 = float(1.5 * (float(abs(t)) ** 3) - 2.5 * (float(abs(t)) ** 2) + 1)
    a2 = float(1.5 * (float(abs(1 - t)) ** 3) - 2.5 * (float(abs(1 - t)) ** 2) + 1)
    a3 = float(
        -0.5 * (float(abs(t + 1)) ** 3) + 2.5 * (float(abs(t + 1)) ** 2) - 4 * abs(t + 1) + 2)
    a4 = float(
        -0.5 * (float(abs(2 - t)) ** 3) + 2.5 * (float(abs(2 - t)) ** 2) - 4 * abs(2 - t) + 2)

    # If all coefficients exceeds 1 - rectify it for eliminating saturation
    if a1 + a2 + a3 + a4 >= 1:
        cubicImage[row][col] = image[row][int(x_original)]
    else:
        cubicImage[row][col] = ((a1 * image[row][int(x_original)]) + (a2 * image[row][int(x_original) + 1])
                                + (a3 * image[row][int(x_original) - 1]) + (a4 * image[row][int(x_original) + 2]))


# Defining the event listener (callback function)
def shape_selection(event, x, y, flags, param):
    # making coordinates global
    global coordinates

    # Storing (x1,y1) coordinates when left mouse button is pressed
    if event == cv.EVENT_LBUTTONDOWN:
        coordinates = [(x, y)]

    # Storing (x2,y2) coordinates when the left mouse button is released and make a rectangle on the selected region
    elif event == cv.EVENT_LBUTTONUP:
        coordinates.append((x, y))

        line_start_point = [int(coordinates[0][0] + (coordinates[1][0] - coordinates[0][0]) / 2), coordinates[0][1]]
        line_end_point = [int(coordinates[0][0] + (coordinates[1][0] - coordinates[0][0]) / 2), coordinates[1][1]]

        image = cv.imread(imageFile)
        # Drawing a rectangle with a middle line around the region of interest (roi)
        cv.rectangle(image, coordinates[0], coordinates[1], yellowColor, line_thickness)
        cv.line(image, line_start_point, line_end_point, yellowColor, line_thickness)
        cv.putText(image, secondInstructionText, textOrigin, font, fontScale, blackColor, text_thickness, cv.LINE_AA)
        cv.imshow(window_name, image)

        while 1:
            keyPressed = cv.waitKey(1) & 0xFF
            # press 'c' to continue
            if keyPressed == ord('c'):
                break
            # press 'q' to exit
            if keyPressed == ord('q'):
                exit()

        image = cv.imread(imageFile)
        # Drawing a rectangle with a middle line around the region of interest (roi)
        cv.rectangle(image, coordinates[0], coordinates[1], yellowColor, line_thickness)
        cv.line(image, line_start_point, line_end_point, yellowColor, line_thickness)
        cv.imshow(window_name, image)

        # wait for clicking on the line to stretch it
        cv.setMouseCallback(window_name, drag_selection)

        while 1:
            keyPressed = cv.waitKey(1) & 0xFF
            # press 'q' to exit
            if keyPressed == ord('q'):
                exit()


# Step 1 - Ask the user to select a ROIc
# Displaying a Text on the image
while 1:
    cv.putText(image, firstInstructionText, textOrigin, font, fontScale, blackColor, text_thickness, cv.LINE_AA)
    cv.imshow(window_name, image)

    keyPressed = cv.waitKey(1) & 0xFF
    # press 'c' to continue
    if keyPressed == ord('c'):
        break
    # press 'q' to exit
    if keyPressed == ord('q'):
        exit()

image = cv.imread(imageFile)
cv.imshow(window_name, image)

# Step 2 - waiting for the user to drag a working rectangle (ROI)
while 1:
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, shape_selection)

    keyPressed = cv.waitKey(1) & 0xFF
    # press 'q' to exit
    if keyPressed == ord('q'):
        exit()
