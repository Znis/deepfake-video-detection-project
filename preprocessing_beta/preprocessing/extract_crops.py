

import argparse
import json
import os
from os import cpu_count
from pathlib import Path
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from functools import partial
from glob import glob
from multiprocessing.pool import Pool

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm



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

def extract_video(param, root_dir, crops_dir):
    video, bboxes_path = param
    with open(bboxes_path, "r") as bbox_f:
        bboxes_dict = json.load(bbox_f)

    capture = cv2.VideoCapture(video)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_dir1 = os.path.join(root_dir,crops_dir, "videos_224")
    vid_dir2 = os.path.join(root_dir,crops_dir, "videos_112")


    for i in range(150):
        capture.grab()
        success, frame = capture.retrieve()
        id = os.path.splitext(os.path.basename(video))[0]
        img_dir = os.path.join(root_dir, crops_dir, id)
        crops = []
        if not success or str(i) not in bboxes_dict:
            continue
        bboxes = bboxes_dict[str(i)]
        if bboxes is None:
            continue
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin
            p_h = h // 3
            p_w = w // 3
            crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            h, w = crop.shape[:2]
            crops.append(crop)
        
        
        os.makedirs(img_dir, exist_ok=True)
        for j, crop in enumerate(crops):
            cv2.imwrite(os.path.join(img_dir, "{}_{}.png".format(i, j)), crop)
    try:    
        video_name1 = vid_dir1 + "/" + id +'.mp4'
        video_name2 = vid_dir2 + "/" + id +'.mp4'

        images = [img for img in os.listdir(img_dir) if img.endswith(".png")]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video1 = cv2.VideoWriter(video_name1, fourcc, 30, (224,224))
        video2 = cv2.VideoWriter(video_name2, fourcc, 30, (112,112))

        for image in images:
            img112 = cv2.resize(cv2.imread(os.path.join(img_dir, image)),(112,112))
            img224 = cv2.resize(cv2.imread(os.path.join(img_dir, image)),(224,224))
            # detection_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img224))
            # print(detection_result)
            video1.write(img224)
            video2.write(img112)
            os.remove(os.path.join(img_dir, image))
        os.rmdir(img_dir)
            

        # cv2.destroyAllWindows()
        video1.release()
        video2.release()
    except Exception as e:
        print(f"Error on {id}.mp4 with Exception: {e}")


# def get_video_paths(root_dir):
#     paths = []
#     for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
#         dir = Path(json_path).parent
#         with open(json_path, "r") as f:
#             metadata = json.load(f)
#         for k, v in metadata.items():
#             original = v.get("original", None)
#             if not original:
#                 original = k
#             bboxes_path = os.path.join(root_dir, "boxes", original[:-4] + ".json")
#             if not os.path.exists(bboxes_path):
#                 continue
#             paths.append((os.path.join(dir, k), bboxes_path))

#     return paths

def get_video_paths(root_dir):
    paths = []
    for vid in glob(os.path.join(root_dir, "*/*.mp4")):    
        dir = Path(vid).parent
        temp = vid.split("/")[-1]
        bboxes_path = os.path.join(root_dir, "boxes", temp[:-4] + ".json")
        if not os.path.exists(bboxes_path):
            continue
        paths.append((os.path.join(dir, temp), bboxes_path))

    return paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extracts crops from video")
    parser.add_argument("--root-dir", help="root directory")
    parser.add_argument("--crops-dir", help="crops directory")

    args = parser.parse_args()
    # base_options = python.BaseOptions(model_asset_path='/home/jenish/dfproj/weights/face_landmarker_v2_with_blendshapes.task' )
    # options = vision.FaceLandmarkerOptions(base_options=base_options,
    #                                    output_face_blendshapes=False,
    #                                    output_facial_transformation_matrixes=False,
    #                                    num_faces=1, min_face_detection_confidence = 0.4, min_face_presence_confidence = 0.4, min_tracking_confidence = 0.4)
    # detector = vision.FaceLandmarker.create_from_options(options)
    os.makedirs(os.path.join(args.root_dir, args.crops_dir), exist_ok=True)
    os.makedirs(os.path.join(args.root_dir, args.crops_dir, "videos_224"), exist_ok=True)
    os.makedirs(os.path.join(args.root_dir, args.crops_dir, "videos_112"), exist_ok=True)
    params = get_video_paths(args.root_dir)
    with Pool(processes=cpu_count()) as p:
        with tqdm(total=len(params)) as pbar:
            for v in p.imap_unordered(partial(extract_video, root_dir=args.root_dir, crops_dir=args.crops_dir), params):
                pbar.update()
