# Mask Detector

This Python class implements a Mask Detector using TensorFlow Lite (TFLite) for inference. The Mask Detector can classify whether a person is wearing a mask, not wearing a mask, or wearing a mask incorrectly.

## Class: MaskDetector

### Methods

#### `__init__(self, model_type=128) -> None`

- Initializes the MaskDetector object with the specified model type.
- Loads the TFLite model based on the model type.
- Initializes labels for the model's output classes.
- Invokes the `load_model()` and `model_warmup()` methods.

#### `load_model(self)`

- Loads the TFLite model specified by the model path.
- Gets input and output details from the model.
- Extracts relevant information such as input and output indices, shapes, and dimensions.

#### `model_stats(self)`

- Prints statistics of the loaded model, including input and output details.

#### `model_warmup(self, warming_steps=10)`

- Warms up the loaded model by invoking it multiple times.
- Helps to ensure consistent performance during inference.

#### `resize_image_keep_aspect_ratio(self, image, target_width, target_height, fill_color=(255, 255, 255)) -> numpy.ndarray`

- Resizes an image while maintaining its aspect ratio.
- Fills the remaining area with a specified fill color.
- Returns the resized image as a numpy array.

#### `predict(self, image) -> str`

- Performs inference on the input image using the loaded model.
- Resizes the input image and prepares input data.
- Sets input tensor, runs inference, and retrieves output tensor.
- Extracts the predicted label based on the output tensor.
- Returns the predicted label as a string.

### Attributes

- `model_path`: Path to the TFLite model file.
- `labels`: Dictionary mapping class indices to class labels.
- `interpreter`: TensorFlow Lite interpreter for model inference.
- `input_index`: Index of the model's input tensor.
- `input_shape`: Shape of the model's input tensor.
- `output_index`: Index of the model's output tensor.
- `output_shape`: Shape of the model's output tensor.
- `input_width`: Width of the model's input tensor.
- `input_height`: Height of the model's input tensor.

## Example Usage

```python
# Create a MaskDetector object
mask_detector = MaskDetector(model_type=192)

# Load an image
image = cv2.imread("test_image.jpg")

# Perform prediction
label = mask_detector.predict(image)
print("Predicted label:", label)
