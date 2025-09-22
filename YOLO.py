import argparse, os, numpy, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import PIL
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
#from yad2k.models.keras_yolo import yolo_head
#from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image

"""
Compute box scores by doing the elementwise product as described in Figure 4 ( 𝑝×𝑐 ).
The following code may help you choose the right operator:

a = rng.standard_normal((19, 19, 5, 1))
b = rng.standard_normal((19, 19, 5, 80))
c = a * b # shape of c will be (19, 19, 5, 80)
This is an example of broadcasting (multiplying vectors of different sizes).

For each box, find:

the index of the class with the maximum box score

the corresponding box score

Useful References

tf.math.argmax
tf.math.reduce_max
Helpful Hints

For the axis parameter of argmax and reduce_max, if you want to select the last axis, one way to do so is to set axis=-1. This is similar to Python array indexing, where you can select the last position of an array using arrayname[-1].
Applying reduce_max normally collapses the axis for which the maximum is applied. keepdims=False is the default option, and allows that dimension to be removed. You don't need to keep the last dimension after applying the maximum here.
Create a mask by using a threshold. As a reminder: ([0.9, 0.3, 0.4, 0.5, 0.1] < 0.4) returns: [False, True, False, False, True]. The mask should be True for the boxes you want to keep.

Use TensorFlow to apply the mask to box_class_scores, boxes and box_classes to filter out the boxes you don't want. You should be left with just the subset of boxes you want to keep.

One more useful reference:

tf.boolean mask
And one more helpful hint: :)

For the tf.boolean_mask, you can keep the default axis=None.
"""
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
"""
The key steps are as follows:

Select the box with the highest score.
Compute the overlap of this box with all other boxes of the same class. Remove any boxes that significantly overlap (IoU >= iou_threshold).
Repeat the process: go back to step 1 and iterate until no remaining boxes have a score lower than the currently selected box.
This process effectively eliminates boxes that overlap significantly with the selected boxes, leaving only the "best" candidates.

For implementation, consider using a divide-and-conquer approach:

Divide all the boxes by class.
Apply non-max suppression for each class individually.
Finally, merge the results from each class into a single set of boxes.
This approach is recommended for the upcoming exercise, as it simplifies the implementation while ensuring accuracy across multiple classes.


Exercise 3 - yolo_non_max_suppression
Implement yolo_non_max_suppression() using TensorFlow. TensorFlow has two built-in functions that are used to implement non-max suppression (so you don't actually need to use your iou() implementation):

Reference documentation:

tf.image.non_max_suppression()
tf.image.non_max_suppression(
  boxes, scores, max_output_size, iou_threshold=0.5,
  score_threshold=float('-inf'), name=None
)
Note: In the exercise below, for tf.image.non_max_suppression() you only need to set boxes, scores, max_output_size and iou_threshold parameters.
tf.gather()
tf.gather(
  params, indices, validate_indices=None, axis=None, batch_dims=0, name=None
)
Note: In the exercise below, for tf.gather() you only need to set params and indices parameters.

tf.boolean mask()
tf.boolean_mask(
  tensor, mask, axis=None, name='boolean_mask'
)
Note: In the exercise below, for tf.boolean_mask() you only need to set tensor and mask parameters.

tf.concat()
tf.concat(
  values, axis, name='concat'
)
"""
def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None, ), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    print(f"\n=== {yolo_non_max_suppression.__name__} ===")
    boxes = tf.cast(boxes, dtype=tf.float32)
    scores = tf.cast(scores, dtype=tf.float32)
    nms_indices = []
    classes_labels = tf.unique(classes)[0] # Get unique classes
    print(f"boxes: {boxes.shape}, scores: {scores.shape}, classes: {classes}, classes_labels: {classes_labels}")

    for label in classes_labels:
        print(f"Process label: {label}...")
        filtering_mask = classes == label
        print(f"filtering_mask: {filtering_mask}")
    
        # Get boxes for this class
        # Use tf.boolean_mask() with 'boxes' and `filtering_mask`
        boxes_label = tf.boolean_mask(boxes, filtering_mask)
        
        # Get scores for this class
        # Use tf.boolean_mask() with 'scores' and `filtering_mask`
        scores_label = tf.boolean_mask(scores, filtering_mask)
        
        if tf.shape(scores_label)[0] > 0:  # Check if there are any boxes to process
            
            # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
            ##(≈ 5 lines)
            nms_indices_label = tf.image.non_max_suppression(
                    boxes_label,
                    scores_label,
                    max_boxes,
                    iou_threshold=iou_threshold) 

            # Get original indices of the selected boxes
            selected_indices = tf.squeeze(tf.where(filtering_mask), axis=1)
            
            # Append the resulting boxes into the partial result
            # Use tf.gather() with 'selected_indices' and `nms_indices_label`
            nms_indices.append(tf.gather(selected_indices, nms_indices_label))

    # Flatten the list of indices and concatenate
    # Use tf.concat() with 'nms_indices' and `axis=0`
    nms_indices = tf.concat(nms_indices, axis=0)
    
    # Use tf.gather() to select only nms_indices from scores, boxes and classes
    ##(≈ 3 lines)
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    
    # Sort by scores and return the top max_boxes
    sort_order = tf.argsort(scores, direction='DESCENDING').numpy()
    scores = tf.gather(scores, sort_order[0:max_boxes])
    boxes = tf.gather(boxes, sort_order[0:max_boxes])
    classes = tf.gather(classes, sort_order[0:max_boxes])

    return scores, boxes, classes

def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])
"""
takes the output of the YOLO encoding and filters the boxes using score threshold and NMS. There's just one last implementational detail you have to know. There're a few ways of representing boxes, such as via their corners or via their midpoint and height/width. YOLO converts between a few such formats at different times, using the following functions (which are provided):

boxes = yolo_boxes_to_corners(box_xy, box_wh) 
which converts the yolo box coordinates (x,y,w,h) to box corners' coordinates (x1, y1, x2, y2) to fit the input of yolo_filter_boxes

boxes = scale_boxes(boxes, image_shape)
YOLO's network was trained to run on 608x608 images. If you are testing this data on a different size image -- for example, the car detection dataset had 720x1280 images -- this step rescales the boxes so that they can be plotted on top of the original 720x1280 image.
"""
def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = float(image_shape[0])
    width = float(image_shape[1])
    image_dims = backend.stack([height, width, height, width])
    image_dims = backend.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def yolo_eval(yolo_outputs, image_shape = (720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    
    ### START CODE HERE
    
    # Retrieve outputs of the YOLO model (≈1 line)
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    
    # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    
    # Use the function `yolo_filter_boxes` you've implemented to perform Score-filtering with a threshold of score_threshold
    scores, boxes, classes = yolo_filter_boxes(boxes, # Use boxes
                                  box_confidence, # Use box confidence
                                  box_class_probs, # Use box class probability
                                  score_threshold  # Use threshold=score_threshold
                                 )
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)
    
    # Use the function `yolo_non_max_suppression` you've implemented to perform Non-max suppression with 
    # maximum number of boxes set to max_boxes and a threshold of iou_threshold
    scores, boxes, classes = yolo_non_max_suppression(scores, # Use scores
                                  boxes, # Use boxes
                                  classes, # Use classes
                                  max_boxes, # Use max boxes
                                  iou_threshold  # Use iou_threshold=iou_threshold
                                 )
    
    ### END CODE HERE
    
    return scores, boxes, classes

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

def yolo_non_max_suppression_test():
    scores = numpy.array([0.855, 0.828])
    boxes = numpy.array([[0.45, 0.2,  1.01, 2.6], [0.42, 0.15, 1.7, 1.01]])
    classes = numpy.array([0, 1])

    print(f"iou:    \t{IoU(boxes[0], boxes[1])}")

    scores2, boxes2, classes2 = yolo_non_max_suppression(scores, boxes, classes, iou_threshold = 0.9)

    assert numpy.allclose(scores2.numpy(), [0.855, 0.828]), f"Wrong value on scores {scores2.numpy()}"
    assert numpy.allclose(boxes2.numpy(), [[0.45, 0.2,  1.01, 2.6], [0.42, 0.15, 1.7, 1.01]]), f"Wrong value on boxes {boxes2.numpy()}"
    assert numpy.array_equal(classes2.numpy(), [0, 1]), f"Wrong value on classes {classes2.numpy()}"

    scores2, boxes2, classes2 = yolo_non_max_suppression(scores, boxes, classes, iou_threshold = 0.1)

    assert numpy.allclose(scores2.numpy(), [0.855, 0.828]), f"Wrong value on scores {scores2.numpy()}"
    assert numpy.allclose(boxes2.numpy(), [[0.45, 0.2,  1.01, 2.6], [0.42, 0.15, 1.7, 1.01]]), f"Wrong value on boxes {boxes2.numpy()}"
    assert numpy.array_equal(classes2.numpy(), [0, 1]), f"Wrong value on classes {classes2.numpy()}"

    classes = numpy.array([0, 0])

    # If both boxes belongs to the same class, one box is suppressed if iou is under the iou_threshold
    scores2, boxes2, classes2 = yolo_non_max_suppression(scores, boxes, classes, iou_threshold = 0.15)

    assert numpy.allclose(scores2.numpy(), [0.855]), f"Wrong value on scores {scores2.numpy()}"
    assert numpy.allclose(boxes2.numpy(), [[0.45, 0.2,  1.01, 2.6]]), f"Wrong value on boxes {boxes2.numpy()}"
    assert numpy.array_equal(classes2.numpy(), [0]), f"Wrong value on classes {classes2.numpy()}"

    # It must return both boxes, as they belong to different classes
    print(f"scores:  \t{scores2.numpy()}")
    print(f"boxes:  \t{boxes2.numpy()}")     
    print(f"classes:\t{classes2.numpy()}")

    # If both boxes belongs to the same class, one box is suppressed if iou is under the iou_threshold
    scores2, boxes2, classes2 = yolo_non_max_suppression(scores, boxes, [0, 0], iou_threshold = 0.9)

    assert numpy.allclose(scores2.numpy(), [0.855, 0.828]), f"Wrong value on scores {scores2.numpy()}"
    assert numpy.allclose(boxes2.numpy(), [[0.45, 0.2,  1.01, 2.6], [0.42, 0.15, 1.7, 1.01]]), f"Wrong value on boxes {boxes2.numpy()}"
    assert numpy.array_equal(classes2.numpy(), [0, 0]), f"Wrong value on classes {classes2.numpy()}"

def yolo_eval_test():
    yolo_outputs = (tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
    scores, boxes, classes = yolo_eval(yolo_outputs)
    print("scores[2] = " + str(scores[2].numpy()))
    print("boxes[2] = " + str(boxes[2].numpy()))
    print("classes[2] = " + str(classes[2].numpy()))
    print("scores.shape = " + str(scores.numpy().shape))
    print("boxes.shape = " + str(boxes.numpy().shape))
    print("classes.shape = " + str(classes.numpy().shape))

    assert type(scores) == EagerTensor, "Use tensoflow functions"
    assert type(boxes) == EagerTensor, "Use tensoflow functions"
    assert type(classes) == EagerTensor, "Use tensoflow functions"

    assert scores.shape == (10,), "Wrong shape"
    assert boxes.shape == (10, 4), "Wrong shape"
    assert classes.shape == (10,), "Wrong shape"

    # The following assertions need np.random.seed(10)
    #assert numpy.isclose(scores[2].numpy(), 171.60194), "Wrong value on scores"
    #assert numpy.allclose(boxes[2].numpy(), [-1240.3483, -3212.5881, -645.78, 2024.3052]), "Wrong value on boxes"
    #assert numpy.isclose(classes[2].numpy(), 16), "Wrong value on classes"    

if __name__ == "__main__":
    yolo_filter_boxes_test()
    IoUTest()
    yolo_non_max_suppression_test()
    yolo_eval_test()