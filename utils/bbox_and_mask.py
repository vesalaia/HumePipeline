"""
Bbox and mask utils

"""
import numpy as np
import shapely
from shapely.ops import linemerge, unary_union, polygonize
#from shapely.validation import make_valid
from shapely.geometry import LineString, Polygon
from shapely.errors import TopologicalError
import copy


def bb_iou(boxA, boxB, Btype=False):
    # determine the (x, y)-coordinates of the intersection rectangle
    if Btype:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
    else:
        xA = max(boxA[0][0], boxB[0][0])
        yA = max(boxA[0][1], boxB[0][1])
        xB = min(boxA[1][0], boxB[1][0])
        yB = min(boxA[1][1], boxB[1][1])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    if Btype:
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    else:
        boxAArea = (boxA[1][0] - boxA[0][0] + 1) * (boxA[1][1] - boxA[0][1] + 1)
        boxBArea = (boxB[1][0] - boxB[0][0] + 1) * (boxB[1][1] - boxB[0][1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def isboxAinsideboxB(boxA, boxB, Btype=False):
    # determine the (x, y)-coordinates of the intersection rectangle
    if Btype:
        x_min_a, y_min_a, x_max_a, y_max_a = boxA
        x_min_b, y_min_b, x_max_b, y_max_b = boxB
    else:
        x_min_a, y_min_a = boxA[0]
        x_max_a, y_max_a = boxA[1]
        x_min_b, y_min_b = boxB[0]
        x_max_b, y_max_b = boxB[1]
        
    return x_min_a >= x_min_b and y_min_a >= y_min_b and x_max_a <= x_max_b and y_max_a <= y_max_b


def mergeBoxes(boxA, boxB, Btype=False):
    if Btype:
        val = [min(boxA[0], boxB[0]),
               min(boxA[1], boxB[1]),
               max(boxA[2], boxB[2]),
               max(boxA[3], boxB[3])]
    else:
        val = [(min(boxA[0][0], boxB[0][0]),
                min(boxA[0][1], boxB[0][1])),
               (max(boxA[1][0], boxB[1][0]),
                max(boxA[1][1], boxB[1][1]))]
        return val

def combineMasks(maskA, maskB):
    poly_A = Polygon(maskA)
    poly_B = Polygon(maskB)
    combined_poly = poly_A.union(poly_B)

    if type(combined_poly) == shapely.geometry.polygon.Polygon:
        return np.array(list(combined_poly.exterior.coords)).astype(int)
    else:  # To do fix if union returns MultiPolygon
        return maskA

def combineMasksIntersection(maskA, maskB):
    poly_A = Polygon(maskA)
    poly_B = Polygon(maskB)
    combined_poly = poly_A.intersection(poly_B)

    if type(combined_poly) == shapely.geometry.polygon.Polygon:
        return np.array(list(combined_poly.exterior.coords)).astype(int)
    else:  # To do fix if union returns MultiPolygon
        return maskA


def check_if_on_same_line(opts, boxA, boxB, Btype=False):
    if Btype:
        minY = min(boxA[1], boxB[1])
        maxY = max(boxA[3], boxB[3])
        heightA = boxA[3] - boxA[1]
        heightB = boxB[3] - boxB[1]
        heightAB = maxY - minY

    else:
        minY = min(boxA[0][1], boxB[0][1])
        maxY = max(boxA[1][1], boxB[1][1])
        heightA = boxA[1][1] - boxA[0][1]
        heightB = boxB[1][1] - boxB[0][1]
        heightAB = maxY - minY
    
    return  ((heightA  + heightA * opts.line_merge_limit / 100) > heightAB) and ((heightB  + heightB * opts.line_merge_limit / 100) > heightAB)

def mergeRegions(opts, boxes, labels, masks, region1, region2, region3):
    #
    # To Do: Masks merging
    #
    #    dist_limit = 0.1
    if opts.DEBUG: 
        print(labels)
        print(boxes)
        print("Labels:", len(labels))
        print("Boxes:", len(boxes))
        print("Masks:", len(masks))
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if j <= i:
                continue

#            if bb_iou(boxes[i], boxes[j]) > 0:
#                print(labels[i],labels[j],boxes[i], boxes[j], bb_iou(boxes[i], boxes[j]))

            if (labels[i] in region1) and (labels[j] in region2) and \
                bb_iou(boxes[i], boxes[j]) >= opts.overlap_threshold: #and \
  #              check_if_on_same_line(opts, boxes[i], boxes[j]):
                print("Regions/Lines merged: ", region1, region2, region3)
                print(labels[i], labels[j], boxes[i], boxes[j], bb_iou(boxes[i],boxes[j]))
                try:
                    boxes[i] = mergeBoxes(boxes[i], boxes[j], False)
                except:
                    print("Exception:", mergeBoxes(boxes[i], boxes[j], False))
                    
                if opts.DEBUG: print("Box1:", boxes[i], "Box1:", boxes[j], "IoU:", bb_iou(boxes[i], boxes[j]))
                labels[i] = region3
                masks[i] = combineMasks(masks[i], masks[j])
                boxes.pop(j)
                labels.pop(j)
                masks.pop(j)
                return True, boxes, labels, masks

    return False, boxes, labels, masks

def mergeOverlappingRegions(opts, boxes, labels, masks):
    #
    # To Do: Masks merging
    #
    #    dist_limit = 0.1
    if opts.DEBUG: 
        print(labels)
        print(boxes)
        print("Labels:", len(labels))
        print("Boxes:", len(boxes))
        print("Masks:", len(masks))
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if j == i:
                continue
            if isboxAinsideboxB(boxes[i], boxes[j]):
                print("Regions/Lines merged: ", labels[i], labels[j], boxes[i], boxes[j])
                try:
                    boxes[j] = mergeBoxes(boxes[i], boxes[j], False)
                except:
                    print("Exception:", mergeBoxes(boxes[i], boxes[j], False))
                    
                masks[j] = combineMasks(masks[i], masks[j])
                boxes.pop(i)
                labels.pop(i)
                masks.pop(i)
                return True, boxes, labels, masks

    return False, boxes, labels, masks
