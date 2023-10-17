import sys

import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread

NUM_OF_FRAMES = 10
STEP = 10
THUMBNAIL_WIDTH = 150
THUMBNAIL_HEIGHT = 150

import numpy
import tqdm


def generate_pixels():
    indnp = numpy.mgrid[0:SLIC_height, 0:SLIC_width].swapaxes(0, 2).swapaxes(0, 1)
    for i in tqdm.tqdm(range(SLIC_ITERATIONS)):
        SLIC_distances = 1 * numpy.ones(image.shape[:2])
        for j in range(SLIC_centers.shape[0]):
            x_low, x_high = int(SLIC_centers[j][3] - step), int(SLIC_centers[j][3] + step)
            y_low, y_high = int(SLIC_centers[j][4] - step), int(SLIC_centers[j][4] + step)

            if x_low <= 0:
                x_low = 0
            # end
            if x_high > SLIC_width:
                x_high = SLIC_width
            # end
            if y_low <= 0:
                y_low = 0
            # end
            if y_high > SLIC_height:
                y_high = SLIC_height
            # end

            cropimg = SLIC_labimg[y_low: y_high, x_low: x_high]
            color_diff = cropimg - SLIC_labimg[int(SLIC_centers[j][4]), int(SLIC_centers[j][3])]
            color_distance = numpy.sqrt(numpy.sum(numpy.square(color_diff), axis=2))

            yy, xx = numpy.ogrid[y_low: y_high, x_low: x_high]
            pixdist = ((yy - SLIC_centers[j][4]) ** 2 + (xx - SLIC_centers[j][3]) ** 2) ** 0.5

            # SLIC_m is "m" in the paper, (m/S)*dxy
            dist = ((color_distance / SLIC_m) ** 2 + (pixdist / step) ** 2) ** 0.5

            distance_crop = SLIC_distances[y_low: y_high, x_low: x_high]
            idx = dist < distance_crop
            distance_crop[idx] = dist[idx]
            SLIC_distances[y_low: y_high, x_low: x_high] = distance_crop
            SLIC_clusters[y_low: y_high, x_low: x_high][idx] = j
        # end

        for k in range(len(SLIC_centers)):
            idx = (SLIC_clusters == k)
            colornp = SLIC_labimg[idx]
            distnp = indnp[idx]
            SLIC_centers[k][0:3] = numpy.sum(colornp, axis=0)
            sumy, sumx = numpy.sum(distnp, axis=0)
            SLIC_centers[k][3:] = sumx, sumy
            SLIC_centers[k] /= numpy.sum(idx)
        # end
    # end



def display_contours(color):
    is_taken = numpy.zeros(image.shape[:2], numpy.bool)
    contours = []

    for i in range(SLIC_width):
        for j in range(SLIC_height):
            nr_p = 0
            for dx, dy in [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]:
                x = i + dx
                y = j + dy
                if x >= 0 and x < SLIC_width and y >= 0 and y < SLIC_height:
                    if is_taken[y, x] == False and SLIC_clusters[j, i] != SLIC_clusters[y, x]:
                        nr_p += 1
                    # end
                # end
            # end

            if nr_p >= 2:
                is_taken[j, i] = True
                contours.append([j, i])
            # end
        # end
    # end
    for i in range(len(contours)):
        image[contours[i][0], contours[i][1]] = color
    # end


# end

def find_local_minimum(center):
    min_grad = 1
    loc_min = center
    for i in range(center[0] - 1, center[0] + 2):
        for j in range(center[1] - 1, center[1] + 2):
            c1 = SLIC_labimg[j + 1, i]
            c2 = SLIC_labimg[j, i + 1]
            c3 = SLIC_labimg[j, i]
            if ((c1[0] - c3[0]) ** 2) ** 0.5 + ((c2[0] - c3[0]) ** 2) ** 0.5 < min_grad:
                min_grad = abs(c1[0] - c3[0]) + abs(c2[0] - c3[0])
                loc_min = [i, j]
            # end
        # end
    # end
    return loc_min


# end

def calculate_centers():
    centers = []
    for i in range(step, SLIC_width - int(step / 2), step):
        for j in range(step, SLIC_height - int(step / 2), step):
            nc = find_local_minimum(center=(i, j))
            color = SLIC_labimg[nc[1], nc[0]]
            center = [color[0], color[1], color[2], nc[0], nc[1]]
            centers.append(center)
        # end
    # end

    return centers


# end


def find_color_dif(super_pixel_1 , super_pixel_2):
    dif_R = abs(super_pixel_1[0]-super_pixel_2[0])/255
    dif_G = abs(super_pixel_1[1] - super_pixel_2[1])/255
    dif_B = abs(super_pixel_1[2] - super_pixel_2[2])/255
    color_dif = 100*(dif_R + dif_G + dif_B)/3
    if color_dif > 1.0:
        color_dif = 0.95
    return color_dif


def find_position_dif(super_pixel_1 , super_pixel_2):
    dif_x = abs(super_pixel_1[3]-super_pixel_2[3])/SLIC_width
    dif_y = abs(super_pixel_1[4]-super_pixel_2[4])/SLIC_height
    position_dif = 1500*(dif_y + dif_x)
    if position_dif > 1.0:
        position_dif = 0.95
    return position_dif


def find_difference(Image1_superpixels, Image2_superpixels):
    diffrence = []
    differences = []
    for i in range(0,len(Image1_superpixels)-1):
        for j in range(0, len(Image2_superpixels)-1):
            color_dif = find_color_dif(Image1_superpixels[i],Image2_superpixels[j])
            position_dif = find_position_dif(Image1_superpixels[i],Image2_superpixels[j])
            found_dif = 0.5*(color_dif + position_dif)
            diffrence.append(found_dif)
        differences.append(min(diffrence))
    average_dif = sum(differences) / len(differences)
    return average_dif


# main
try:
    videoFile = sys.argv[1]
except IndexError:
    videoFile = 'elephants.mp4'


# receiving the video and creating Frames array
frames = []
video_capture = cv2.VideoCapture(videoFile)

success, image = video_capture.read()

count = 0
length = 0

while success:
    if count % STEP == 0 and length < NUM_OF_FRAMES:
        name = "frames/frame%d.jpg" % length
        # cv2.imshow(name, image)
        # wait 1 sec. only for debugging - remove before deploying
        # cv2.waitKey(1000)
        frames.append(image)
        cv2.imwrite(name, image)
        length += 1
    success, image = video_capture.read()
    count += 1

# creating the superpixels and finding the similarity
SLIC_centers =[]
previous_centers = []
current_centers=[]
similarity_array =[]
dif =0.0
SLIC_m = int(40)
SLIC_ITERATIONS = 4

for i in range(len(frames)):
    if i == 0:
        image = frames[i]
        step = int((image.shape[0] * image.shape[1] / 100) ** 0.5)
        SLIC_height, SLIC_width = image.shape[:2]
        SLIC_labimg = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(numpy.float64)
        SLIC_distances = 1 * numpy.ones(image.shape[:2])
        SLIC_clusters = -1 * SLIC_distances
        SLIC_center_counts = numpy.zeros(len(calculate_centers()))
        SLIC_centers = numpy.array(calculate_centers())

        generate_pixels()
        current_centers = calculate_centers()


    else:
        image = frames[i]
        step = int((image.shape[0] * image.shape[1] / 100) ** 0.5)
        SLIC_height, SLIC_width = image.shape[:2]
        SLIC_labimg = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(numpy.float64)
        SLIC_distances = 1 * numpy.ones(image.shape[:2])
        SLIC_clusters = -1 * SLIC_distances
        SLIC_center_counts = numpy.zeros(len(calculate_centers()))
        SLIC_centers = numpy.array(calculate_centers())

        generate_pixels()
        previous_centers = current_centers
        current_centers = calculate_centers()
        dif = find_difference(previous_centers, current_centers)
        similarity_array.append(1-dif)

# Graph construction

left = list(range(1, len(similarity_array)+1))
height = similarity_array

plt.bar(left, height, width=0.8, color=['blue'])

# naming the x-axis
plt.xlabel('Frame number')
# naming the y-axis
plt.ylabel('Similarity')
# plot title
plt.title('Similarity between two consecutive frames')
plt.xticks(left)

# function to show the plot
plt.show()
print('The similarities are:',similarity_array)
