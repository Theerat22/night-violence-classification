import streamlit as st
import numpy as np
import cv2
import os
import subprocess
import tensorflow
from collections import deque
from tensorflow.keras.models import load_model

model = load_model('Mobilenet_modelV2.h5')

SEQUENCE_LENGTH = 16
IMAGE_HEIGHT,IMAGE_WIDTH = 64,64
CLASSES_LIST = ["Violence","NonViolence"]

def predict_frames(video_file_path, output_file_path, SEQUENCE_LENGTH):

    video_reader = cv2.VideoCapture(video_file_path)

    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                    video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    frames_queue = deque(maxlen = SEQUENCE_LENGTH)

    predicted_class_name = ''

    while video_reader.isOpened():

        ok, frame = video_reader.read()

        if not ok:
            break

        # Resize
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:

            # Predict
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis = 0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]

        # Write classname
        if predicted_class_name == "Violence":
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 12)
        else:
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 12)

        # Write into the disk
        video_writer.write(frame)

    video_reader.release()
    video_writer.release()


def main():
    st.sidebar.title("About")

    st.sidebar.info("""
    Machine Learning for predict Night Violence Video 
    By Theeratdolchat Chatchai
    """)
    st.sidebar.page_link("https://medium.com/@sitthach7777/night-violence-classification-ตรวจจับความรุนแรงในยามวิกาล-ef980f9de419", label="Medium", icon="🌎")
    st.sidebar.page_link("https://drive.google.com/file/d/1XcVAO-kDQ4uXvDOPHhasy9uRKM5Uk4pg/view?usp=sharing", label="Notebook", icon="🌙")
    st.sidebar.page_link("https://github.com/Theerat22/night-violence-classification.git", label="Github", icon="🌟")
    st.sidebar.page_link("https://drive.google.com/file/d/1mE9muV_ZfgemjrEmkr4Cl_jQgt_p-4it/view?usp=sharing", label="How to use", icon="❔")

    st.title('Night Violence Classification - ตรวจจับความรุนแรงในยามวิกาล')
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg","mov","avi"])
    if uploaded_file is not None:
        upload_name = "playback/temp_video.mp4"
        with open(upload_name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File Uploaded Successfully!")

        if st.button('Classify'):
            output_video = 'playback/playback.mp4'
            with st.spinner('Wait for it...'):
                predict_frames(upload_name,output_video,SEQUENCE_LENGTH)
                st.success('Done!')
                st.video(output_video)
    else:
        st.subheader("Please upload a video file.")

if __name__ == '__main__':
    main()