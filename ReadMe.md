# Real-time Face Mask Detection Project

## Introduction
This project implements real-time face detection and mask detection using TensorFlow Lite for face detection and a custom trained mask detection model. It detects faces in images or video streams in real-time, draws bounding boxes around the detected faces, and predicts whether the person is wearing a mask or not.

## Files
- **main.py**: This is the main Python script containing the implementation of the face and mask detection system.
- **BlazeFaceDetection/**:
  - **BlazeFace.py**: This file contains the classes and functions for face detection, including anchor generation, detection post-processing, and TensorFlow Lite model loading.
- **Mask_Detector/**:
  - **MaskDetector.py**: This file contains the MaskDetector class for mask detection, including model loading and prediction.
- **Models/**:
  - **face_detection_front.tflite**: TensorFlow Lite model file for face detection (BlazeFace) with a shorter range.
  - **face_detection_full_range.tflite**: TensorFlow Lite model file for face detection (BlazeFace) with the full detection range.
  - **face_detection_back.tflite**: TensorFlow Lite model file for face detection (BlazeFace) with the back camera configuration.
  - **mask_detection_128.tflite**: TensorFlow Lite model file for mask detection with a image size of 128.
  - **mask_detection_192.tflite**: TensorFlow Lite model file for mask detection with a image size of 192.

## Dependencies
- TensorFlow
- OpenCV
- NumPy

## Usage
1. Clone the repository: `git clone https://github.com/your-username/real-time-face-mask-detection.git`
2. Navigate to the project directory: `cd real-time-face-mask-detection`
3. Install the dependencies: `pip install -r requirements.txt`
4. Run the main script: `python main.py`

## References
- [BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs](https://arxiv.org/abs/1907.05047) by Valentin Bazarevsky and Andrei Tkachenka.
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [OpenCV Documentation](https://docs.opencv.org/)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
