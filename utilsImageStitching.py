import os, sys
import cv2
import random
from heapq import nsmallest
import numpy as np
import matplotlib.pyplot as plt
from detectBlobs import DetectBlobs
# from detectBlobsSolution import DetectBlobs

# detectKeypoints(...): Detect feature keypoints in the input image
#   You can either reuse your blob detector from part 1 of this assignment
#   or you can use the provided compiled blob detector detectBlobsSolution.pyc
#
#   Input:
#        im  - input image
#   Output:
#        detected feature points (in any format you like).

def detectKeypoints(im):
    # YOUR CODE STARTS HERE
    # im = im/255
    detected_blobs = DetectBlobs(im/255)
    print ('Detect %d blobs' % (detected_blobs.shape[0]))
    return detected_blobs




# computeDescriptors(...): compute descriptors from the detected keypoints
#   You can build the descriptors by flatting the pixels in the local
#   neighborhood of each keypoint, or by using the SIFT feature descriptors from
#   OpenCV (see computeSIFTDescriptors(...)). Use the detected blob radii to
#   define the neighborhood scales.
#
#   Input:
#        im          - input image
#        keypoints   - detected feature points
#
#   Output:
#        descriptors - n x dim array, where n is the number of keypoints
#                      and dim is the dimension of each descriptor.
#
def computeDescriptors(im, keypoints):
    # YOUR CODE STARTS HERE
    return computeSIFTDescriptors(im, keypoints)
    # return np.zeros(shape=(len(keypoints['pt']), 10), dtype=np.float32)


# computeSIFTDescriptors(...): compute SIFT feature descriptors from the
#   detected keypoints. This function is provided to you.
#
#   Input:
#        im          - H x W array, the input image
#        keypoints   - n x 4 array, where there are n blobs detected and
#                      each row is [x, y, radius, score]
#
#   Output:
#        descriptors - n x 128 array, where n is the number of keypoints
#                      and 128 is the dimension of each descriptor.
#
def computeSIFTDescriptors(im, keypoints):
    kp = []
    for blob in keypoints:
        kp.append(cv2.KeyPoint(blob[1], blob[0], _size=blob[2]*2, _response=blob[-1], _class_id=len(kp)))
    detector = cv2.xfeatures2d_SIFT.create()
    return detector.compute(im, kp)[1]



# getMatches(...): match two groups of descriptors.
#
#   There are several strategies you can use to match keypoints from the left
#   image to those in the right image. Feel free to use any (or combinations
#   of) strategies:
#
#   - Return all putative matches. You can select all pairs whose
#   descriptor distances are below a specified threshold,
#   or select the top few hundred descriptor pairs with the
#   smallest pairwise distances.
#
#   - KNN-Match. For each keypoint in the left image, you can simply return the
#   the K best pairings with keypoints in the right image.
#
#   - Lowe's Ratio Test. For each pair of keypoints to be returned, if the
#   next best match with the same left keypoint is nearly as good as the
#   current match to be returned, then this match should be thrown out.
#   For example, given point A in the left image and its best and second best
#   matches B and C in the right image, we check: distance(A,B) < distance(A,C)*0.75
#   If this test fails, don't return pair (A,B)
#
#
#   Input:
#         descriptors1 - the descriptors of the first image
#         descriptors2 - the descriptors of the second image
#
#   Output:
#         index1       - 1-D array contains the indices of descriptors1 in matches
#         index2       - 1-D array contains the indices of descriptors2 in matches

def getMatches(descriptors1, descriptors2):
    # YOUR CODE STARTS HERE
    # Goal 1: Use top 250 points
    # Goal 2: Use Lowe's

    counter = 0
    index1 = []
    index2 = []
    temp_storage = []
    lowest = 5000000
    second_lowest = 6000000
    stored_ind = 0
    stored = False

    # From 3 ways to compute distances
    normd1 = np.sum(descriptors1**2,axis=1,keepdims=True)
    normd2 = np.sum(descriptors2**2,axis=1,keepdims=True)
    D3 = (normd1+normd2.T-2*np.dot(descriptors1,descriptors2.T))**0.5

    for d1 in range(len(descriptors1)):
        lowest = min(D3[d1])
        stored_ind = np.argmin(D3[d1])
        second_lowest = nsmallest(2, D3[d1])[-1]
        if lowest <= second_lowest * 0.80: #0.8 for hill
            if counter > 224:
                idx = temp_storage.index(max(temp_storage))
                index1[idx] = d1
                index2[idx] = stored_ind
                temp_storage[idx] = lowest
            else:
                counter += 1
                index1.append(d1)
                index2.append(stored_ind)
                temp_storage.append(lowest)

    # print(len(index2))
    return np.array(index1), np.array(index2)




# RANSAC(...): run the RANSAC algorithm to estimate a homography mapping between two images.
#   Input:
#        matches - two 1-D arrays that contain the indices on matches.
#        keypoints1       - keypoints on the left image
#        keypoints2       - keypoints on the right image
#
#   Output:
#        H                - 3 x 3 array, a homography mapping between two images
#        numInliers       - int, the number of inliers
#
#   Note: Use four matches to initialize the homography in each iteration.
#         You should output a single transformation that gets the most inliers
#         in the course of all the iterations. For the various RANSAC parameters
#         (number of iterations, inlier threshold), play around with a few
#         "reasonable" values and pick the ones that work best.

def RANSAC(matches, keypoints1, keypoints2):
    s = 4
    N = 2500 #3000 #10000

    H = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]

    max_inliers = 0
    my_inliers = []

    bestH = [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]

    bestA = []

    for i in range(N):
        idx = random.sample(range(len(matches[0])), s)
        idx1 = []
        idx2 = []
        for i in idx:
            idx1.append(matches[0][i])
            idx2.append(matches[1][i])

        p1 = keypoints1[idx1[0]]
        pp1 = keypoints2[idx2[0]]
        p2 = keypoints1[idx1[1]]
        pp2 = keypoints2[idx2[1]]
        p3 = keypoints1[idx1[2]]
        pp3 = keypoints2[idx2[2]]
        p4 = keypoints1[idx1[3]]
        pp4 = keypoints2[idx2[3]]

        A = np.array([RANSACfit(p1, pp1)[0], RANSACfit(p1, pp1)[1],
                      RANSACfit(p2, pp2)[0], RANSACfit(p2, pp2)[1],
                      RANSACfit(p3, pp3)[0], RANSACfit(p3, pp3)[1],
                      RANSACfit(p4, pp4)[0], RANSACfit(p4, pp4)[1]])

        AT = A.transpose()
        ATA = np.matmul(AT, A)

        evals, evecs = np.linalg.eig(ATA)
        lowest_eval_idx = np.argmin(evals)
        lowest_eval_evec = evecs[:,lowest_eval_idx]

        H = lowest_eval_evec.reshape((3, 3))

        num_inliers = 0
        new_inliers = []
        for match in range(len(matches[0])):
            if match in idx:
                continue
            else:
                k1 = keypoints1[matches[0][match]]
                k2 = keypoints2[matches[1][match]]
                orig_point = np.array((k1[1], k1[0], 1))
                new_point = np.matmul(H, orig_point)
                new_x = new_point[0]/new_point[2]
                new_y = new_point[1]/new_point[2]
                pred_p = ((new_x, new_y))
                act_p = np.array((k2[1], k2[0]))

                if np.linalg.norm(act_p - pred_p) <= 2:
                    new_inliers.append(match)
                    num_inliers += 1
        if (num_inliers >= max_inliers):
            max_inliers = num_inliers
            bestH = H
            bestA = A
            my_inliers = new_inliers

    to_add = []
    # counter = 0
    for inlier in my_inliers:
        to_add.append(RANSACfit(keypoints1[matches[0]][inlier], keypoints2[matches[1]][inlier])[0])
        to_add.append(RANSACfit(keypoints1[matches[0]][inlier], keypoints2[matches[1]][inlier])[1])
        # counter += 1

    newA = np.array(to_add)

    finalA = np.concatenate((bestA, newA))

    finalAT = finalA.transpose()
    finalATA = np.matmul(finalAT, finalA)

    f_evals, f_evecs = np.linalg.eig(finalATA)
    f_lowest_eval_idx = np.argmin(f_evals)
    f_lowest_eval_evec = f_evecs[:,f_lowest_eval_idx]
    # print(lowest_eval_evec)

    H = f_lowest_eval_evec.reshape((3, 3))

    srcP = []
    dstP = []

    num_inliers = 0
    new_inliers = []
    for m in range(len(matches[0])):
        # print(matches)
        k1 = keypoints1[matches[0][m]]
        # srcP.append([k1[1], k1[0]])
        k2 = keypoints2[matches[1][m]]
        # dstP.append([k2[1], k2[0]])
        orig_point = np.array((k1[1], k1[0], 1))
        new_point = np.matmul(H, orig_point)
        new_x = new_point[0]/new_point[2]
        new_y = new_point[1]/new_point[2]
        pred_p = ((new_x, new_y))
        act_p = np.array((k2[1], k2[0]))
        if np.linalg.norm(act_p - pred_p) <= 3: #5?
            new_inliers.append(m)
            num_inliers += 1

    # srcP = np.array(srcP)
    # dstP = np.array(dstP)
    # cH, mask = cv2.findHomography(srcP, dstP)
    # print("CH: ")
    # print(cH)

    # print("final num inliers:")
    # print(num_inliers)

    return np.array(H), num_inliers


def RANSACfit(p1, pp1):
    return ([0, 0, 0, p1[1], p1[0], 1, -pp1[0]*p1[1], -pp1[0]*p1[0], -pp1[0]],
            [p1[1], p1[0], 1, 0, 0, 0, -pp1[1]*p1[1], -pp1[1]*p1[0], -pp1[1]])


# warpImageWithMapping(...): warp one image using the homography mapping and
#   composite the warped image and another image into a panorama.
#
#   Input:
#        im_left, im_right - input images.
#        H                 - 3 x 3 array, a homography mapping
#
#   Output:
#        Panorama made of the warped image and the other.
#
#       To display the full warped image, you may want to modify the matrix H.
#       CLUE: first get the coordinates of the corners under the transformation,
#             use the new corners to determine the offsets to move the
#             warped image such that it can be displayed completely.
#             Modify H to fulfill this translate operation.
#       You can use cv2.warpPerspective(...) to warp your image using H

def warpImageWithMapping(im_left, im_right, H):
    # YOUR CODE STARTS HERE
    # NOTE: warp left image onto right image

    (h1, w1) = im_left.shape
    (h2, w2) = im_right.shape
    # print((h1, w1))

    corners = []
    corners.append(warpPoint([0, 0], H))
    corners.append(warpPoint([0, h1-1], H))
    corners.append(warpPoint([w1-1, 0], H))
    corners.append(warpPoint([w1-1, h1-1], H))
    corners = np.array(corners)


    corners_x = corners[:,0]
    corners_y = corners[:,1]

    xmax = np.max(corners_x)
    xmin = np.min(corners_x)
    ymax = np.max(corners_y)
    ymin = np.min(corners_y)

    if (ymin > 0):
        # ymax += ymin
        ymin = 0
    if (xmin > 0):
        # xmax += xmin
        xmin = 0

    t = [-xmin, -ymin]

    T = np.array([[1, 0, t[0]],
         [0, 1, t[1]],
         [0, 0, 1]])

    H1 = np.matmul(T, H)

    totw = w1 + w2
    toth = h1 + h2
    xmin = int(xmin)
    ymin = int(ymin)

    warpw = int(xmax-xmin)
    warph = int(ymax-ymin)
    totalheight = max(warph, im_right.shape[0]-int(ymin))
    totalwidth = max(warpw, im_right.shape[1]-int(xmin))

    new_image = np.zeros((totalheight, totalwidth), dtype=np.uint8)

    warped = cv2.warpPerspective(im_left, H1, (warpw, warph))
    new_image[0:warph, 0:warpw] = warped
    new_image[-int(ymin):im_right.shape[0]-int(ymin), -int(xmin):im_right.shape[1]-int(xmin)] = im_right

    return new_image, np.linalg.inv(T), np.linalg.inv(H1)
    # return warped


def warpPoint(pt, H):

    new_pt = np.matmul(H, np.array((pt[0], pt[1], 1)))
    return [new_pt[0]/new_pt[2], new_pt[1]/new_pt[2]]

    # print(H[2])
    # lam = H[2][0]*pt[0] + H[2][1]*pt[1] + H[2][2]
    # xp = (H[0][0]*pt[0] + H[0][1]*pt[1] + H[0][2])/lam
    # yp = (H[1][0]*pt[0] + H[1][1]*pt[1] + H[1][2])/lam
    # return [xp, yp]

# drawMatches(...): draw matches between the two images and display the image.
#
#   Input:
#         im1: input image on the left
#         im2: input image on the right
#         matches: (1-D array, 1-D array) that contains indices of descriptors in matches
#         keypoints1: keypoints on the left image
#         keypoints2: keypoints on the right image
#         title: title of the displayed image.
#
#   Note: This is a utility function that is provided to you. Feel free to
#   modify the code to adapt to the keypoints and matches in your own format.

def trimBlackParts(im):
    (h, w) = im.shape
    # print((h, w))

    toCutCols = []
    toCutCols2 = []
    toCutRows = []
    toCutRows2 = []

    allBlack = False

    toggled = False
    for i in range(h):
        allBlack = True
        for j in range(w):
            if im[i][j] > 0:
                if allBlack:
                    toggled = True
                allBlack = False
        if allBlack:
            if toggled:
                toCutRows2.append(i)
            else:
                toCutRows.append(i)

    toggled = False
    for k in range(w):
        allBlack = True
        for m in range(h):
            if im[m][k] > 0:
                if allBlack:
                    toggled = True
                allBlack = False
        if allBlack:
            if toggled:
                toCutCols2.append(k)
            else:
                toCutCols.append(k)

    if len(toCutRows2) > 0:
        im = np.delete(im, toCutRows2, axis=0)
    if len(toCutRows) > 0:
        im = np.delete(im, toCutRows, axis=0)

    if len(toCutCols2) > 0:
        im = np.delete(im, toCutCols2, axis=1)
    if len(toCutCols) > 0:
        im = np.delete(im, toCutCols, axis=1)




    # for row in reversed(toCutRows):
    #     im = np.delete(im, row, 1)
    #
    # for col in reversed(toCutCols):
    #     im = np.delete(im, col, 0)

    return im

def drawMatches(im1, im2, matches, keypoints1, keypoints2, title='matches'):
    idx1, idx2 = matches

    cv2matches = []
    for i,j in zip(idx1, idx2):
        cv2matches.append(cv2.DMatch(i, j, _distance=0))

    _kp1, _kp2 = [], []
    for i in range(len(keypoints1)):
        _kp1.append(cv2.KeyPoint(keypoints1[i][1], keypoints1[i][0], _size=keypoints1[i][2], _response=keypoints1[i][3], _class_id=len(_kp1)))
    for i in range(len(keypoints2)):
        _kp2.append(cv2.KeyPoint(keypoints2[i][1], keypoints2[i][0], _size=keypoints2[i][2], _response=keypoints2[i][3], _class_id=len(_kp2)))

    # for i in range(len(keypoints1['pt'])):
    #     _kp1.append(cv2.KeyPoint(keypoints1['pt'][i][1], keypoints1['pt'][i][0], _size=keypoints1['radius'][i], _response=keypoints1['score'][i], _class_id=len(_kp1)))
    # for i in range(len(keypoints2['pt'])):
    #     _kp2.append(cv2.KeyPoint(keypoints2['pt'][i][1], keypoints2['pt'][i][0], _size=keypoints2['radius'][i], _response=keypoints2['score'][i], _class_id=len(_kp2)))

    im_matches = np.empty((max(im1.shape[0], im2.shape[0]), im1.shape[1]+im2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(im1, _kp1, im2, _kp2, cv2matches, im_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, im_matches)
    # cv2.waitKey()
