import cv2
from BlazeFaceDetection.BlazeFace import FaceDetectionModule, FaceDetectionModelType
from Mask_Detector.MaskDetector import MaskDetector


def main():
    """
    Main function to perform face detection and mask detection on an image.
    """

    # Initialize the face detection module
    face_detector = FaceDetectionModule(FaceDetectionModelType.BLAZEFACE_BACK, 0.5)
    mask_detector = MaskDetector(model_type=192)
    
    # Load image using OpenCV
    img_path = "faces.jpg"
    image = cv2.imread(img_path)

    # Perform inference
    detections = face_detector.detect_from_image(image)
    print("Number of faces detected:", len(detections))

    if detections:
        height, width = image.shape[:2]
        _max_side = max(width, height)
        r_height = max(0, (width-height)//2)
        r_width = max(0, (height-width)//2)

        # Draw bounding boxes on the original image and perform mask detection
        for det in detections:
            x1 = int(_max_side*det.xmin) - r_width
            y1 = int(_max_side*det.ymin) - r_height
            x2 = int(_max_side*(det.xmin + det.width)) - r_width 
            y2 = int(_max_side*(det.ymin + det.height)) - r_height
            
            face = image[y1:y2, x1:x2]
            mask_label = mask_detector.predict(face)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (200, 12, 12), 2)
            
            # Put text on the image
            cv2.putText(image, mask_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 12, 12), 2)

    # Save the result
    cv2.imwrite("detected_faces.jpg", image)

if __name__ == "__main__":
    main()
