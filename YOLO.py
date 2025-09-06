import argparse, os, numpy, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import PIL
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from tensorflow.keras.models import load_model
#from yad2k.models.keras_yolo import yolo_head
#from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image

# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: yolo_filter_boxes
def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = .6):
    print(f"\n=== {yolo_filter_boxes.__name__} ===")
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
        boxes -- tensor of shape (19, 19, 5, 4) containing the midpoint and dimensions (𝑏𝑥,𝑏𝑦,𝑏ℎ,𝑏𝑤) for each of the 5 boxes in each cell.
        box_confidence -- tensor of shape (19, 19, 5, 1) containing 𝑝𝑐 (confidence probability that there's some object) for each of the 5 boxes predicted in each of the 19x19 cells.
        box_class_probs -- tensor of shape (19, 19, 5, 80) containing the "class probabilities" (𝑐1,𝑐2,...𝑐80) for each of the 80 classes for each of the 5 boxes per cell.
        threshold -- real value, if [ highest class probability score < threshold],
                     then get rid of the corresponding box

    Returns:
        scores -- tensor of shape (None,), containing the class probability score for selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    # Step 1: Compute box scores
    ##(≈ 1 line)
    box_scores = box_confidence * box_class_probs
    #print(f"box_scores: {box_scores.shape}")
    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    ##(≈ 2 lines)
    # IMPORTANT: set axis to -1
    box_classes = numpy.argmax(box_scores, axis=-1)
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1) #, keepdims=True)
    #print(f"box_classes: {box_classes.shape}")
    #print(box_classes[0,0,0])
    #print(f"box_class_scores: {box_class_scores.shape}")
    #print(box_class_scores[0])
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    ## (≈ 1 line)
    filtering_mask = (box_class_scores >= threshold)
    #print(f"filtering_mask: {filtering_mask.shape}")
    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    ## (≈ 3 lines)
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    #print(f"scores: {scores.shape}")
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    return scores, boxes, classes

def IoU(box1, box2):
    print(f"\n=== {IoU.__name__} ===")
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    ### START CODE HERE
    inter_area = 0
    if box1_x1 <= box2_x1 and box1_y1 <= box2_y1:
        inter_width = box1_x2 - box2_x1
        inter_height = box1_y2 - box2_y1
        if inter_width >= 0 and inter_height >= 0:
            inter_area = inter_width * inter_height
    elif box1_x1 >= box2_x1 and box1_y1 <= box2_y1:
        inter_width = box2_x2 - box1_x1
        inter_height = box1_y2 - box2_y1
        if inter_width >= 0 and inter_height >= 0:
            inter_area = inter_width * inter_height       
    elif box2_x1 <= box1_x1 and box2_y1 <= box1_y1:
        inter_width = box2_x2 - box1_x1
        inter_height = box2_y2 - box1_y1
        if inter_width >= 0 and inter_height >= 0:
            inter_area = inter_width * inter_height
    elif box2_x1 >= box1_x1 and box2_y1 <= box1_y1:
        inter_width = box1_x2 - box2_x1
        inter_height = box2_y2 - box1_y1
        if inter_width >= 0 and inter_height >= 0:
            inter_area = inter_width * inter_height       
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ## (≈ 3 lines)
    box1_area = (box1_x2 - box1_x1) * (box1_y2- box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2- box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    # compute the IoU
    iou = inter_area / union_area
    ### END CODE HERE
    
    return iou

def yolo_filter_boxes_test():
    box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
    boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
    box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = 0.5)

    print("scores[2] = " + str(scores[2].numpy()))
    print("boxes[2] = " + str(boxes[2].numpy()))
    print("classes[2] = " + str(classes[2].numpy()))
    print("scores.shape = " + str(scores.shape))
    print("boxes.shape = " + str(boxes.shape))
    print("classes.shape = " + str(classes.shape))

def IoUTest():
    box1 = (2, 1, 4, 3)
    box2 = (1, 2, 3, 4)

    print("iou for intersecting boxes = " + str(IoU(box1, box2)))
    assert IoU(box1, box2) < 1, "The intersection area must be always smaller or equal than the union area."
    assert numpy.isclose(IoU(box1, box2), 0.14285714), "Wrong value. Check your implementation. Problem with intersecting boxes"

    ## Test case 2: boxes do not intersect
    box1 = (1,2,3,4)
    box2 = (5,6,7,8)
    print("iou for non-intersecting boxes = " + str(IoU(box1,box2)))
    assert IoU(box1, box2) == 0, "Intersection must be 0"

    ## Test case 3: boxes intersect at vertices only
    box1 = (1,1,2,2)
    box2 = (2,2,3,3)
    print("iou for boxes that only touch at vertices = " + str(IoU(box1,box2)))
    assert IoU(box1, box2) == 0, "Intersection at vertices must be 0"

    ## Test case 4: boxes intersect at edge only
    box1 = (1,1,3,3)
    box2 = (2,3,3,4)
    print("iou for boxes that only touch at edges = " + str(IoU(box1,box2)))
    assert IoU(box1, box2) == 0, "Intersection at edges must be 0"    
    
if __name__ == "__main__":
    yolo_filter_boxes_test()
    IoUTest()