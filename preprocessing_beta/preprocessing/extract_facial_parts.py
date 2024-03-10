
import os
import shutil
from tqdm import tqdm
import glob
import cv2
import numpy as np
import itertools
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_face_mesh = mp.solutions.face_mesh
base_options = python.BaseOptions(model_asset_path='/home/jenish/dfproj/weights/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                    output_face_blendshapes=False,
                                    output_facial_transformation_matrixes=False,
                                    num_faces=1, min_face_detection_confidence = 0.01, min_face_presence_confidence = 0.01, min_tracking_confidence = 0.01)

detector = vision.FaceLandmarker.create_from_options(options)





    



def main():
    
    os.makedirs("/home/jenish/dfproj/facial_parts_dset0", exist_ok=True)
    vid_list = glob.glob("/home/jenish/dfproj/preprocessing_beta/data_root/dfdc_train_part_19"+"/*.mp4")
    vid_list.sort()
    extract_facial_parts(vid_list)



if __name__ == '__main__':
    main()
