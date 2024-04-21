# BlazeFace Detection Module

This Python file contains the implementation of a face detection module using the BlazeFace model.

## Contents

1. **Configurations**: Definitions of minimum detection score threshold and NMS threshold.

2. **SSD Anchors Calculator Options Class**: Class to calculate SSD anchors options used in the BlazeFace model.

3. **Anchor Class**: Class representing an anchor.

4. **IndexedScore Class**: Class representing an indexed score.

5. **Detection Class**: Class representing a detection.

6. **TensorsToDetectionsCalculatorOptions Class**: Class representing options for converting model tensors to detections.

7. **FaceDetectionModelType Enum Class**: Enumeration class representing BlazeFace model types.

8. **FaceDetectionModule Class**: Class implementing the face detection module.

## Class: SSD Anchors Calculator Options

### Attributes

- `input_size_width` (int): Width of input images.
- `input_size_height` (int): Height of input images.
- `min_scale` (float): Minimum scale for generating anchor boxes on feature maps.
- `max_scale` (float): Maximum scale for generating anchor boxes on feature maps.
- `anchor_offset_x` (float): Offset for the center of anchors along the x-axis.
- `anchor_offset_y` (float): Offset for the center of anchors along the y-axis.
- `aspect_ratios` (list): List of different aspect ratios to generate anchors.
- `interpolated_scale_aspect_ratio` (float): Additional anchor added with this aspect ratio.
- `reduce_boxes_in_lowest_layer` (bool): Indicates whether fixed 3 boxes per location is used in the lowest layer.
- `fixed_anchor_size` (bool): Indicates whether to use fixed width and height for each anchor.
- `strides` (list): Strides of each output feature map.
- `num_layers` (int): Number of output feature maps to generate the anchors on.
- `aspect_ratios_size` (int): Size of aspect ratio to generate anchors.

### Methods

- `to_str()`: Converts object attributes to a string.

## Class: Anchor

### Attributes

- `x_center` (float): X coordinate of the anchor's center.
- `y_center` (float): Y coordinate of the anchor's center.
- `h` (float): Height of the anchor.
- `w` (float): Width of the anchor.

### Methods

- `to_str()`: Converts object attributes to a string.

## Class: IndexedScore

### Attributes

- `index` (int): Index associated with the score.
- `score` (float): The score value.

## Class: Detection

### Attributes

- `score` (float): Confidence score of the detection.
- `class_id` (float): Class identifier of the detected object.
- `xmin` (float): X-coordinate of the top-left corner of the bounding box.
- `ymin` (float): Y-coordinate of the top-left corner of the bounding box.
- `width` (float): Width of the bounding box.
- `height` (float): Height of the bounding box.
- `landmark_points` (list): List of landmark points.

### Methods

- `to_str()`: Converts object attributes to a string.

## Class: TensorsToDetectionsCalculatorOptions

### Attributes

- `num_classes` (int): Number of classes.
- `num_boxes` (int): Number of boxes.
- `num_coords` (int): Number of coordinates.
- `keypoint_coord_offset` (int): Offset for keypoint coordinates.
- `num_keypoints` (int): Number of keypoints.
- `num_values_per_keypoint` (int): Number of values per keypoint.
- `box_coord_offset` (int): Offset for box coordinates.
- `x_scale` (float): Scaling factor for x-coordinate.
- `y_scale` (float): Scaling factor for y-coordinate.
- `w_scale` (float): Scaling factor for width.
- `h_scale` (float): Scaling factor for height.
- `score_clipping_thresh` (float): Threshold for clipping scores.
- `min_score_thresh` (float): Minimum score threshold.
- `apply_exponential_on_box_size` (bool): Whether to apply exponential on box size.
- `reverse_output_order` (bool): Whether to reverse the output order.
- `sigmoid_score` (bool): Whether to apply sigmoid on scores.
- `flip_vertically` (bool): Whether to flip vertically.

### Methods

- `to_str()`: Converts object attributes to a string.

## Class: FaceDetectionModelType Enum

Enumeration class representing BlazeFace model types.

## Class: FaceDetectionModule

Class implementing the face detection module.

### Attributes

- `_BLAZEFACE_SHORT_MODEL_PATH` (str): Path to the BlazeFace short model.
- `_BLAZEFACE_FULL_MODEL_PATH` (str): Path to the BlazeFace full model.
- `_BLAZEFACE_BACK_MODEL_PATH` (str): Path to the BlazeFace back model.

### Methods

- `__init__(self, model_type=FaceDetectionModelType.BLAZEFACE_FULL, detection_threshold=MIN_DETECTION_SCORE_THRESHOLD)`: Initializes the face detection module.
- `load_model_weights(self)`: Loads the model weights based on the selected model type.
- `detect_from_image` : detect the face from given single image(array format).
- `Some Other Helping Funtion`: Other function they provide for code maintaining.
