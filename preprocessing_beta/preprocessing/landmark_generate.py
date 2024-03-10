
import os

import shutil
from tqdm import tqdm
import glob

import cv2

import numpy as np

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision





# #draw function
# def draw_landmarks_on_image(rgb_image, detection_result):
#   face_landmarks_list = detection_result.face_landmarks
#   annotated_image = np.copy(rgb_image)

#   # Loop through the detected faces to visualize.
#   for idx in range(len(face_landmarks_list)):
#     face_landmarks = face_landmarks_list[idx]

#     # Draw the face landmarks.
#     face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     face_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
#     ])

#     solutions.drawing_utils.draw_landmarks(
#         image=annotated_image,
#         landmark_list=face_landmarks_proto,
#         connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp.solutions.drawing_styles
#         .get_default_face_mesh_tesselation_style())
#     solutions.drawing_utils.draw_landmarks(
#         image=annotated_image,
#         landmark_list=face_landmarks_proto,
#         connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp.solutions.drawing_styles
#         .get_default_face_mesh_contours_style())
#     solutions.drawing_utils.draw_landmarks(
#         image=annotated_image,
#         landmark_list=face_landmarks_proto,
#         connections=mp.solutions.face_mesh.FACEMESH_IRISES,
#           landmark_drawing_spec=None,
#           connection_drawing_spec=mp.solutions.drawing_styles
#           .get_default_face_mesh_iris_connections_style())

#   return annotated_image

# def plot_face_blendshapes_bar_graph(face_blendshapes):
#   # Extract the face blendshapes category names and scores.
#   face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
#   face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
#   # The blendshapes are ordered in decreasing score value.
#   face_blendshapes_ranks = range(len(face_blendshapes_names))

#   fig, ax = plt.subplots(figsize=(12, 12))
#   bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
#   ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
#   ax.invert_yaxis()

#   # Label each bar with values
#   for score, patch in zip(face_blendshapes_scores, bar.patches):
#     plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

#   ax.set_xlabel('Score')
#   ax.set_title("Face Blendshapes")
#   plt.tight_layout()
#   plt.show()











def extract_landmarks(vid_list ):

    for i,vid in enumerate(tqdm(vid_list)):
        if i % 500 == 0 and i != 0:
          shutil.move("/home/jenish/dfproj/landmarks_dset0", f"/home/jenish/dfproj/landmark_checkpoints_dset0/checkpoint_{i/500}" )
          os.makedirs("/home/jenish/dfproj/landmarks_dset0", exist_ok=True)
        ori_name = vid.split("/")[-1]
        ori_name = ori_name.split(".")[0]
        ori_dir = os.path.join("/home/jenish/dfproj/landmarks_dset0")
        landmark_dir = os.path.join(ori_dir,ori_name)
        os.makedirs(landmark_dir, exist_ok=True)
        capture = cv2.VideoCapture(vid)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(50):
            capture.grab()
            success, frame = capture.retrieve()
            if not success :
                continue
            
            detection_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
            try:
                face_landmarks =  detection_result.face_landmarks[0]
                
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
                        ])
            
                frame_landmarks = np.array([])
            
                for landmark in face_landmarks_proto.landmark:
                    
                    landmark_x = np.around(landmark.x, decimals=8)
                    landmark_y = np.around(landmark.y, decimals=8)
                    landmark_z = np.around(landmark.z, decimals=8)
                    arr = np.array([landmark_x, landmark_y, landmark_z])
                    frame_landmarks = np.append(frame_landmarks, arr, axis=0)
                landmark_path = os.path.join(landmark_dir, f"{i+1}")
                np.save(landmark_path, frame_landmarks)
    
            except Exception as e:
                pass

        

        

            










def main():
    base_options = python.BaseOptions(model_asset_path='/home/jenish/dfproj/weights/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=False,
                                        output_facial_transformation_matrixes=False,
                                        num_faces=1, min_face_detection_confidence = 0.01, min_face_presence_confidence = 0.01, min_tracking_confidence = 0.01)
    global detector
    detector = vision.FaceLandmarker.create_from_options(options)
    os.makedirs("/home/jenish/dfproj/landmarks_dset0", exist_ok=True)
    vid_list = glob.glob("/home/jenish/dfproj/preprocessing_beta/data_root/dfdc_train_part_19"+"/*.mp4")
    vid_list.sort()
    extract_landmarks(vid_list)



if __name__ == '__main__':
    main()
