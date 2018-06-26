# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
cimport numpy as np

cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
    return a if a <= b else b

def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
    """
    the function here accepts the array [boxes , scores] and thresh value
    """

    #store the x and y coordinates into x1,y1 and x2,y2 variables
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]

    #store the scores into scores variable
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]

    #find the area of the the box
    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    #sort the scores and return the sorted index based on the scores
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]

    #ndets contains the total number of boxes passed , i.e the number of rows
    cdef int ndets = dets.shape[0]

    #make a numpy.zeros array of size ndets , i.e with size equal to number of rows
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j

    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix1, iy1, ix2, iy2, iarea

    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx1, yy1, xx2, yy2
    cdef np.float32_t w, h
    cdef np.float32_t inter, ovr

    keep = []

    #iterate over all the boxes and keep only those boxes which have IOU below certain threshold
    #take a box and iterate over the others and compare their IOUs
    #remove the boxes which overlap the same object by considering the pre defined threshold
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)

        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue

            #maybe faulty code here
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])

            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)

            inter = w * h

            #finding the IOU value
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                #if over a certain threshold then supress those bounding boxes as they are pointing to the same object
                suppressed[j] = 1

    #return only those indexes which we have to keep , the rest are supressed
    return keep
