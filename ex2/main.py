import sys
import cv2 as cv

try:
    imageFile = sys.argv[1]
except IndexError:
    raise SystemExit(f"usage: {sys.argv[0]} <name_of_image_file>")

# load image as grayscale
image = cv.imread(imageFile, cv.IMREAD_GRAYSCALE)

# Step 1: Image Binarization
_, image_result = cv.threshold(image, 200, 255, cv.THRESH_BINARY_INV)

# Step 2: find contours - Component extraction
contours, hierarchy = cv.findContours(image_result, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print("number of contours:", len(contours))


def is_patah(contour):
    x, y, w, h = cv.boundingRect(contour)
    if w < h:
        return bool(False)
    return bool(True)


def is_kamatz(contour):
    x, y, w, h = cv.boundingRect(contour)
    counter = 0
    for i in range(x, x + w):
        for j in range(y, y + h):
            if image[j][i] == 255:
                counter += 1
    return bool(counter > (h * w / 4))


def delete_contour(contour_threshold, round):
    for contour in contours:
        if cv.contourArea(contour) < contour_threshold:
            if round == 1:
                cv.drawContours(image, [contour], 0, (255), -1)
            elif round == 2:
                if is_patah(contour):
                    cv.drawContours(image, [contour], 0, (255), -1)
            else:
                if is_patah(contour) and is_kamatz(contour):
                    cv.drawContours(image, [contour], 0, (255), -1)


# create a list of the indexes of the contours and their sizes
contour_sizes = []
for index, contour in enumerate(contours):
    contour_sizes.append([index, cv.contourArea(contour)])

# sort the list based on the contour size.
# this changes the order of the elements in the list
contour_sizes.sort(key=lambda x: x[1])

# contour_threshold = contour_sizes[(int)(indexOfMaxDifference)][1]
delete_contour(0.01 * contour_sizes[-1][1], 1)
delete_contour(0.05 * contour_sizes[-1][1], 2)
delete_contour(0.09 * contour_sizes[-1][1], 3)

# display the result
cv.imshow('result', image)
cv.waitKey(0)
cv.destroyAllWindows()
