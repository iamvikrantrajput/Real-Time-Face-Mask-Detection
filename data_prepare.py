import cv2
from pathlib import Path
import numpy as np
import pandas as pd
from BlazeFaceDetection.BlazeFace import *  

def crop_faces(image, detections):
    height, width = image.shape[:2]
    _max_side = max(width, height)
    r_height = max(0, (width-height)//2)
    r_width = max(0, (height-width)//2)
    faces = []
    # Draw bounding boxes on the original image
    pad = 5
    for det in detections:
        x1 = max(0, int(_max_side*det.xmin) - r_width - pad)
        y1 = max(0, int(_max_side*det.ymin) - r_height - pad)
        x2 = min(_max_side, int(_max_side*(det.xmin + det.width)) - r_width + pad)
        y2 = min(_max_side, int(_max_side*(det.ymin + det.height)) - r_height + pad)
        
        face = image[y1:y2, x1:x2]
        faces.append(face)
    return faces

def generate_df(image_dir, label):
    filepath = pd.Series(list(image_dir.glob(r'*jpg')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=filepath.index).astype(str)
    df = pd.concat([filepath, labels], axis=1)
    return df

def get_data():
    with_mask_dir_s = Path('FMD_DATASET/with_mask/simple/')
    with_mask_dir_c = Path('FMD_DATASET/with_mask/complex/')

    without_mask_dir_s = Path('FMD_DATASET/without_mask/simple/')
    without_mask_dir_c = Path('FMD_DATASET/without_mask/complex/')

    mc_dir = Path('FMD_DATASET/incorrect_mask/mc/')
    mmc_dir = Path('FMD_DATASET/incorrect_mask/mmc/')

    with_mask_dir_s.glob(r'*jpg')
    with_mask_dir_c.glob(r'*jpg')

    without_mask_dir_s.glob(r'*jpg')
    without_mask_dir_c.glob(r'*jpg')

    mc_dir.glob(r'*jpg')
    mmc_dir.glob(r'*jpg')

    with_mask_df_s = generate_df(with_mask_dir_s, label=0)
    with_mask_df_c = generate_df(with_mask_dir_c, label=0)

    without_mask_df_s = generate_df(without_mask_dir_s, label=1)
    without_mask_df_c = generate_df(without_mask_dir_c, label=1)

    mc_df = generate_df(mc_dir, label=2)
    mmc_df = generate_df(mmc_dir, label=2)

    total_df = pd.concat([with_mask_df_s, with_mask_df_c, without_mask_df_s, without_mask_df_c, mc_df, mmc_df], axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)
    return total_df

def main():

    # Initialize the face detection module
    face_detector = FaceDetectionModule(FaceDetectionModelType.BLAZEFACE_BACK, 0.6)
    data_df = get_data()
    # Load image using OpenCV
    for img_path in data_df['Filepath']:
        img_save_path = img_path.replace("FMD_DATASET", "PREPARE_DATA")
        image = cv2.imread(img_path)

        # Perform inference
        detections = face_detector.detect_from_image(image)
        print("Number of faces detected:", len(detections))
        if detections:
            faces = crop_faces(image, detections)

            for i, face in enumerate(faces):
                cv2.imwrite(img_save_path.replace(".jpg", f"_{i}.jpg"), face)
        else:
            cv2.imwrite(img_save_path, image)

if __name__ == "__main__":
    main()