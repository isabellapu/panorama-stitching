import numpy as np
import math
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

# This code is taken and converted to Python from:
#
#   CMPSCI 670: Computer Vision, Fall 2014
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
# Part1:
#
#   DetectBlobs(...) detects blobs in the image using the Laplacian
#   of Gaussian filter. Blobs of different size are detected by scaling sigma
#   as well as the size of the filter or the size of the image. Downsampling
#   the image will be faster than upsampling the filter, but the decision of
#   how to implement this function is up to you.
#
#   For each filter scale or image scale and sigma, you will need to keep track of
#   the location and matching score for every blob detection. To combine the 2D maps
#   of blob detections for each scale and for each sigma into a single 2D map of
#   blob detections with varying radii and matching scores, you will need to use
#   Non-Max Suppression (NMS).
#
#   Additional Notes:
#       - We greyscale the input image for simplicity
#       - For a simple implementation of Non-Max-Suppression, you can suppress
#           all but the most likely detection within a sliding window over the
#           2D maps of blob detections (ndimage.maximum_filter may help).
#           To combine blob detections into a single 2D output,
#           you can take the max along the sigma and scale axes. If there are
#           still too many blobs detected, you can do a final NMS. Remember to
#           keep track of the blob radii.
#       - A tip that may improve your LoG filter: Normalize your LoG filter
#           values so that your blobs detections aren't biased towards larger
#           filters sizes
#
#   You can qualitatively evaluate your code using the evalBlobs.py script.
#
# Input:
#   im             - input image
#   sigma          - base sigma of the LoG filter
#   num_intervals  - number of sigma values for each filter size
#   threshold      - threshold for blob detection
#
# Ouput:
#   blobs          - n x 4 array with blob in each row in (x, y, radius, score)
#
def DetectBlobs(
    im,
    sigma = 2,
    num_intervals = 12,
    threshold = 1e-7 #0.33 #1e-4
    ):

    # Convert image to grayscale and convert it to double [0 1].
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)/255

    # YOUR CODE STARTS HERE
    if (im.shape[0] * im.shape[1]) < 300*300:
        window_size = 5
    elif (im.shape[0] * im.shape[1]) < 500*500:
        window_size = 7
    elif (im.shape[0] * im.shape[1]) < 800*800:
        window_size = 9
    else:
        window_size = 11

    k = 2**(1/float(num_intervals))
    k_exp = -1
    laplacian = CreateFilter(sigma, k, k_exp);
    new_im = ndimage.convolve(im, laplacian)
    saved_im = new_im**2

    num_octaves = 4
    blobs = []
    (h, w) = im.shape
    scalespace = np.zeros((h,w,(num_octaves*(num_intervals+2))));
    radius = np.zeros((h,w,(num_octaves*(num_intervals+2))));

    im_copy = np.copy(im)

    for octave in range(num_octaves):
        k = 2**(1/float(num_intervals))
        k_exp = -1
        if octave != 0:
            im_copy = cv2.resize(im_copy, (0,0), fx=0.5, fy=0.5)
        (h2, w2) = im_copy.shape
        for int in range(num_intervals+2):
            laplacian = CreateFilter(sigma, k, k_exp);
            new_im = ndimage.convolve(im_copy, laplacian)
            new_new_im = new_im**2
            first_nms = ndimage.maximum_filter(new_new_im, 2*np.floor(sigma)+1)
            maxes = (new_new_im - first_nms)
            new_new_im[new_new_im < threshold] = 0
            for i in range(h2-1):
                for j in range(w2-1):
                    if maxes[i, j] >= 0:
                        scalespace[i*(2**octave), j*(2**octave), (octave*(num_intervals+2))+int] = new_new_im[i, j]
                        radius[i*(2**octave), j*(2**octave), (octave*(num_intervals+2))+int] = sigma*(k**k_exp)*math.sqrt(2)*(2**octave)
            k_exp += 1

    final_im = np.zeros((h, w))
    radiuses = np.zeros((h, w))

    for i in range(h-1):
        for j in range(w-1):
            score = np.max(scalespace[i, j, :])
            n = np.argmax(scalespace[i, j, :])
            if score > 0:
                final_im[i, j] = score
                radiuses[i, j] = radius[i, j, n]
            else:
                final_im[i, j] = saved_im[i, j]
                radiuses[i, j] = 0

    third_nms = ndimage.maximum_filter(final_im, window_size) #11 for tower, 7 for hill
    help = (final_im - third_nms)
    for i in range(h-1):
        for j in range(w-1):
            if help[i, j] >= 0 and radiuses[i,j] > 0:
                blob = [i, j, radiuses[i, j], 1]
                blobs.append(blob)

    blobs = np.array(blobs)
    return blobs

def CreateFilter(
    sigma=2,
    k=0.1,
    k_exp=1
    ):

    j = k**(k_exp)
    k = k**(k_exp+1)

    filter_size = round(3*sigma)*2+1
    assert filter_size % 2 == 1
    cp = filter_size // 2
    gaussian = np.zeros((filter_size, filter_size))
    gaussian2 = np.zeros((filter_size, filter_size))
    laplacian = np.zeros((filter_size, filter_size))
    sum = 0
    sum2 = 0

    for x in range(filter_size):
        for y in range(filter_size):
            gaussian[x, y] = (1/(2*np.pi*(k*sigma)**2))*np.exp(-((x-cp)**2 + (y-cp)**2)/(2 * (k*sigma)**2))
            num = np.exp(-((x-cp)**2 + (y-cp)**2)/(2 * (j*sigma)**2))
            gaussian2[x, y] = num/(2*np.pi*(j*sigma)**2)
            sum += gaussian[x, y]
            sum2 += gaussian2[x, y]
            # laplacian[x, y] = gaussian[x, y] - gaussian2[x, y]

    gaussian /= sum
    gaussian2 /= sum2
    #
    for x in range(filter_size):
        for y in range(filter_size):
            laplacian[x, y] = gaussian[x, y] - gaussian2[x, y]

    return laplacian


def show(img):
    fig,ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')
    plt.show()
