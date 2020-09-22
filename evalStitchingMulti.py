import os, sys
import cv2
import numpy as np
from heapq import nlargest
from utilsImageStitching import *

imagePath = sys.argv[1]

images = []
for fn in os.listdir(imagePath):
    images.append(cv2.imread(os.path.join(imagePath, fn), cv2.IMREAD_GRAYSCALE)) #cv2.IMREAD_COLOR))

# Build your strategy for multi-image stitching.
# For full credit, the order of merging the images should be determined automatically.
# The basic idea is to first run RANSAC between every pair of images to determine the
# number of inliers to each transformation, use this information to determine which
# pair of images should be merged first (and of these, which one should be the "source"
# and which the "destination"), merge this pair, and proceed recursively.

# YOUR CODE STARTS HERE

kp = []
# matches = np.zeros((len(images), len(images)))
mats = []
mats_nums = []
descr = []

num_im = len(images)

for im in images:
    kp.append(detectKeypoints(im))

# kp = np.array(kp)

for im, k in zip(images, kp):
    descr.append(computeDescriptors(im, k))

mats = np.zeros(((num_im, num_im)), dtype=object)
mats_nums = np.zeros((num_im, num_im))

for d in range(num_im):
    for d2 in range(num_im):
        if d <= d2: continue
        matches = getMatches(descr[d], descr[d2])
        mats[d, d2] = matches
        mats[d2, d] = matches
        mats_nums[d, d2] = len(matches[0])
        mats_nums[d2, d] = len(matches[0])


irrelevant = []

for pic in range(num_im):
    if max(mats_nums[pic]) < 30:
        irrelevant.append(pic)

H_matrices = np.zeros((num_im, num_im), dtype=np.ndarray)
inliers = np.zeros((num_im, num_im))

for d in range(num_im):
    if d in irrelevant: continue
    for d2 in range(num_im):
        if d2 in irrelevant: continue
        if d <= d2: continue
        if mats_nums[d][d2] < 30: continue
        H, numInliers = RANSAC(mats[d, d2], kp[d], kp[d2])
        inliers[d][d2] = numInliers
        inliers[d2][d] = numInliers
        H_matrices[d][d2] = H
        H_matrices[d2][d] = np.linalg.inv(H)


num_ops = (num_im - len(irrelevant) - 1)

new_im = images[0].copy
order = []
rights = set()


max_in = 0
max_coord = (0, 0)
solitary = False

#check if any columns or rows only have 1 non-zero number
for x in range(len(H_matrices)):
    largest = max(inliers[x])
    second_largest = nlargest(2, inliers[x])[-1]
    if second_largest == 0 and largest != 0:
        solitary = True
        max_coord = (np.argmax(inliers[x]), x)
        # print("it worked!")
        # print(max_coord)
        # print(np.argmax(inliers[x]))

if not solitary:
    for x in range(len(H_matrices)):
        for y in range(len(H_matrices[0])):
            if inliers[x][y] > max_in:
                max_in = inliers[x][y]
                max_coord = (x, y)

order.append(max_coord)
inliers[max_coord] = 0
inliers[max_coord[1]][max_coord[0]] = 0
rights.add(max_coord[0])

for i in range(num_ops - 1):
    myCoord = order[-1]
    myX = order[-1][1]
    myY = order[-1][0]
    max_in = 0
    max_coord = (0, 0)
    for i in range(len(inliers[myX])):
        if i in rights: continue
        if inliers[myX][i] > max_in:
            max_in = inliers[myX][i]
            max_coord = (myX, i)
    if max_in == 0:
        for i in range(len(inliers[myY])):
            # print(inliers[myY][i])
            if i in rights: continue
            if inliers[myY][i] > max_in:
                max_in = inliers[myY][i]
                max_coord = (myY, i)
    inliers[max_coord] = 0
    inliers[max_coord[1]][max_coord[0]] = 0
    rights.add(max_coord[1])
    order.append(max_coord)

imCurrent = images[order[0][0]]
H0 = [[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]]
T0 = np.array(H0)
H0 = np.array(H0)
old_coord = [-1, -1]

for coord in order:
    imCurrent = trimBlackParts(imCurrent)
    if old_coord[0] < 0:
        # print("1st step")
        imCurrent, T0, H0 = warpImageWithMapping(imCurrent, images[coord[1]], H_matrices[coord[0], coord[1]])
        old_coord = coord

    elif old_coord[0] == coord[0]:
        # print("option 1")
        imCurrent, T0, H0 = warpImageWithMapping(imCurrent, images[coord[1]], np.dot(H_matrices[coord[0], coord[1]], H0))
        old_coord = coord
    elif old_coord[1] == coord[0]:
        # print("option 2")
        imCurrent, T0, H0 = warpImageWithMapping(imCurrent, images[coord[1]], np.dot(H_matrices[coord[0], coord[1]], T0))
        old_coord = coord
    else:
        # print("option 3?")
        imCurrent, T0, H0 = warpImageWithMapping(imCurrent, images[coord[1]], np.dot(H_matrices[coord[0], coord[1]], T0))
        old_coord = coord

# print("finishing")

imCurrent = trimBlackParts(imCurrent)



cv2.imwrite(sys.argv[2], imCurrent)

cv2.imshow('Panorama', imCurrent)

cv2.waitKey()
