import streamlit as st
import numpy as np
import cv2
import os
import subprocess
import tensorflow
from collections import deque
from tensorflow.keras.models import load_model

MODEL = load_model('Mobilenet_modelV2.h5')

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
            predicted_labels_probabilities = MODEL.predict(np.expand_dims(frames_queue, axis = 0))[0]
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
    st.title('Night Violence Classification')
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg"])
    if uploaded_file is not None:
        st.success("File Uploaded Successfully")
        if st.button('Classify The Video'):
            output_video = 'playback.mp4'
            with st.spinner('Wait for it...'):
                predict_frames(uploaded_file,output_video,SEQUENCE_LENGTH)
                st.success('Done!')
        st.video(output_video)
    else:
        st.text("Please upload a video file")

if __name__ == '__main__':
    main()