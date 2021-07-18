import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import face_recognition

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

model_face_mesh = mp_face_mesh.FaceMesh()

st.title("OpenCV Operations")
st.subheader("Image operations")


st.write("This application performs various operations with OpenCV")


add_selectbox = st.sidebar.selectbox(
    "What operations you would like to perform?",
    ("About", "Face Recognition",'Face Detection','selfie Segmentation')
)

if add_selectbox == "About":
    st.write("This application is a demo for streamlit.")
elif add_selectbox == "Face Detection":
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils 
    

    image_file_path = st.sidebar.file_uploader("Upload image")
    image_train = face_recognition.load_image_file(image_file_path)
    image_encodings_train = face_recognition.face_encodings(image_train)[0]
    image_location_train = face_recognition.face_locations(image_train)[0]
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results:
                image_train = image
                cv2.rectangle(image_train, 
                    (image_location_train[3], image_location_train[0]),
                    (image_location_train[1], image_location_train[2]),
                    (0, 255, 0),
                    2)
                st.image(image) 
            else:
                st.write(f"Could not recognize the face. Result was {results}")
elif add_selectbox == "Face Recognition":
    image_file_path = st.sidebar.file_uploader("Upload image")
    image_file_path_2 = st.sidebar.file_uploader("Upload  test image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        image_train = face_recognition.load_image_file(image_file_path)
        image_encodings_train = face_recognition.face_encodings(image_train)[0]
        image_location_train = face_recognition.face_locations(image_train)[0]
        image_2 = np.array(Image.open(image_file_path_2))
        image_test = face_recognition.load_image_file(image_file_path_2)
        image_encodings_test = face_recognition.face_encodings(image_test)[0]
        results = face_recognition.compare_faces([image_encodings_test], image_encodings_train)[0]
        dst = face_recognition.face_distance([image_encodings_train],image_encodings_test)
        if results:
            image_train = cv2.cvtColor(image_train, cv2.COLOR_BGR2RGB)
            cv2.rectangle(image_train, 
                (image_location_train[3], image_location_train[0]),
                (image_location_train[1], image_location_train[2]),
                (0, 255, 0),
                2)
            cv2.putText(image_train,f"{results} {dst}",
                (60, 60),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255,0),
                1)
            image_color =  cv2.cvtColor(image_train, cv2.COLOR_BGR2RGB)
            st.image(image_color)
        else:
           st.write(f"Could not recognize the face. Result was {results} and distance was {dst}")


elif add_selectbox == "selfie Segmentation":
    mp_drawing = mp.solutions.drawing_utils
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    colour = st.sidebar.radio("choose background colour for your image",
     ('blue', 'green', "red",'black',"white"))
    if colour == "blue":
      BG_COLOR = (0,0, 255)
    elif colour == "green":
        BG_COLOR = (0,255, 0)
    elif colour == "black":
        BG_COLOR = (0,0,0)
    elif colour == "red":
        BG_COLOR = (255,0,0)
    elif colour == "white":
        BG_COLOR = (255,255, 255)          

    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=0) as selfie_segmentation:
          image_file_path = st.sidebar.file_uploader("Upload image")
          if image_file_path is not None:
              image = np.array(Image.open(image_file_path))
              st.sidebar.image(image)
              results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
              # Draw selfie segmentation on the background image.
              # To improve segmentation around boundaries, consider applying a joint
              # bilateral filter to "results.segmentation_mask" with "image".
              condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
              # Generate solid color images for showing the output selfie segmentation mask.
              bg_image = np.zeros(image.shape, dtype=np.uint8)
              bg_image[:] = BG_COLOR
              output_image = np.where(condition, image, bg_image)
              st.image(output_image)       
