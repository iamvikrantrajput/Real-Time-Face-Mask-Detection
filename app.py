import cv2
from BlazeFaceDetection.BlazeFace import FaceDetectionModule, FaceDetectionModelType
from Mask_Detector.MaskDetector import MaskDetector

LABEL_COLOR = {'Without Mask': (35, 7, 245), "With Mask": (47, 245, 7), "Incorrect Mask": (7, 245, 241)}

def main():
    """
    Main function to perform face detection and mask detection on camera input.
    """

    # Initialize the face detection module
    face_detector = FaceDetectionModule(FaceDetectionModelType.BLAZEFACE_BACK, 0.5)
    mask_detector = MaskDetector(model_type=128)

    # Open the default camera
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    _max_side = max(width, height)
    r_height = max(0, (width-height)//2)
    r_width = max(0, (height-width)//2)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Horizontally flip the frame
        frame = cv2.flip(frame, 1)

        # Perform face detection
        detections = face_detector.detect_from_image(frame)
        # print("Number of faces detected:", len(detections))

        if detections:            

            # Draw bounding boxes on the original image and perform mask detection
            pad = 5
            for det in detections:
                x1 = max(0, int(_max_side*det.xmin) - r_width - pad)
                y1 = max(0, int(_max_side*det.ymin) - r_height - pad)
                x2 = min(_max_side, int(_max_side*(det.xmin + det.width)) - r_width + pad)
                y2 = min(_max_side, int(_max_side*(det.ymin + det.height)) - r_height + pad)
                
                face = frame[y1:y2, x1:x2]
                mask_label = mask_detector.predict(face)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), LABEL_COLOR[mask_label], 2)
                
                # Put text on the image
                cv2.putText(frame, mask_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_COLOR[mask_label], 2)

        
        
        # Display the resulting frame
        cv2.imshow('Face Mask Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
