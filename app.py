import streamlit as st
import numpy as np
import cv2
from collections import deque
from tensorflow.keras.models import load_model


model = load_model('Mobilenet_modelV3.h5')

SEQUENCE_LENGTH = 16
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
CLASSES_LIST = ["Violence", "NonViolence"]

def predict_webcam(SEQUENCE_LENGTH):
    video_reader = cv2.VideoCapture(1)  

    if not video_reader.isOpened():
        st.error("Could not open webcam.")
        return

    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''
    stframe = st.empty()

    while True:
        ok, frame = video_reader.read()

        if not ok:
            st.error("Could not read frame from webcam.")
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]

        if predicted_class_name == "Violence":
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 12)
        else:
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 12)


        stframe.image(frame, channels="BGR")

        if st.button('Stop'):
            break

    video_reader.release()

def main():
    st.sidebar.title("About")

    st.sidebar.info("""
    Machine Learning for predicting Night Violence using a webcam.
    By Theeratdolchat Chatchai
    """)
    st.sidebar.page_link("https://medium.com/@sitthach7777/night-violence-classification-ตรวจจับความรุนแรงในยามวิกาล-ef980f9de419", label="Medium", icon="🌎")
    st.sidebar.page_link("https://drive.google.com/file/d/1XcVAO-kDQ4uXvDOPHhasy9uRKM5Uk4pg/view?usp=sharing", label="Notebook", icon="🌙")
    st.sidebar.page_link("https://github.com/Theerat22/night-violence-classification.git", label="Github", icon="🌟")
    st.sidebar.page_link("https://drive.google.com/file/d/1mE9muV_ZfgemjrEmkr4Cl_jQgt_p-4it/view?usp=sharing", label="How to use", icon="❔")

    st.title('Night Violence Classification - Webcam')

    if st.button('Start'):
        st.info('Webcam is starting. Please wait...')
        predict_webcam(SEQUENCE_LENGTH)

if __name__ == '__main__':
    main()
