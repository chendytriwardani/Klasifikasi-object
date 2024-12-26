import streamlit as st
from stqdm import stqdm

import keras
from keras import layers
from keras.utils import load_img, img_to_array, to_categorical
from keras.applications.mobilenet_v2 import MobileNetV2

import cv2
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

le = LabelEncoder()


def record_video(input_kelas, count_recordframe):
    st.markdown(
        "<h3>Capture Frame for Class: %s</h3>" % input_kelas,
        unsafe_allow_html=True,
    )

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("http://192.168.1.17:8080/video")
    fps = 12
    shut_speed = 1 / fps
    frames = []
    class_images = []
    temp = 0

    cv2.namedWindow("Frame Capture")
    start = True
    while start:
        temp += 1
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(
            frame,
            " Kelas: %s - Sample: %.f" % (input_kelas, temp),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        # mirror image
        cv2.imshow("Frame Capture", frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
        class_images.append(input_kelas)
        time.sleep(shut_speed)

        if cv2.waitKey(1) & 0xFF == ord("q") or temp == count_recordframe:
            start = False

    cv2.destroyAllWindows()

    data_images = np.array(frames)
    class_images = np.array(class_images)

    # show first frame, middle frame, last frame
    st.markdown("<h3>Sample Frame</h3>", unsafe_allow_html=True)
    st.image(data_images[[0, len(data_images) // 2, -1]])
    st.markdown(
        "<h5> Total Sample Frame: {}</h5>".format(data_images.shape[0]),
        unsafe_allow_html=True,
    )
    cap.release()
    st.success("Frame Capture for Class: {}!".format(input_kelas), icon="✔️")

    # st.write(data_images.shape, class_images.shape)
    return data_images, class_images


def get_ImagesClassForm():
    session_input = list(st.session_state)

    key_class_input = []
    key_images_input = []
    for input in session_input:
        if "class_input" in input:
            key_class_input.append(input)
        elif "image_input" in input:
            key_images_input.append(input)
    key_class_input = sorted(key_class_input)
    key_images_input = sorted(key_images_input)

    data_images = []
    class_images = []

    for data_img, cls_img in zip(key_images_input, key_class_input):
        kelas = st.session_state[cls_img]
        image = st.session_state[data_img]

        if image:
            for img_input in image:
                # Add class
                class_images.append(kelas)
                # Add Image
                img = load_img(img_input, target_size=(224, 224))
                data_images.append(img_to_array(img))

    for i in range(len(key_class_input)):
        recorded_frames_key = "recorded_frames{}".format(i)
        recorded_classes_key = "recorded_class{}".format(i)
        # st.write(recorded_frames_key, recorded_classes_key)
        if (
            recorded_frames_key in st.session_state
            and recorded_classes_key in st.session_state
        ):
            recorded_frames = st.session_state[recorded_frames_key]
            recorded_classes = st.session_state[recorded_classes_key]

            data_images.extend(recorded_frames)
            class_images.extend(recorded_classes)
    # cek unique class
    if len(np.unique(class_images)) < 2:
        return False

    data_images = np.array(data_images) / 255.0
    class_images = np.array(class_images)
    # st.write("gabung: ", data_images.shape, class_images.shape)

    return data_images, class_images


def trainingModel(epochs, batch_size):
    epochs = epochs
    batch_size = batch_size
    global glob_input_kelas, glob_path_model, glob_str_kelas, glob_y_test_pred_class, globy_y_test_class

    # data input
    X, y = get_ImagesClassForm()
    # st.write(y)
    y_num = le.fit_transform(y)

    glob_input_kelas = le.classes_
    # st.write(glob_input_kelas)
    glob_str_kelas = "-".join(glob_input_kelas)

    # one hot encoding
    y_cat = to_categorical(y_num)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42
    )
    # Model
    base_model = MobileNetV2(
        weights="imagenet", input_shape=X_train[0].shape, include_top=False
    )
    base_model.trainable = False
    inputs = keras.Input(shape=X_train[0].shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(16, activation="relu")(x)

    outputs = layers.Dense(len(y_cat[0]), activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    # Train model

    # progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in stqdm(range(epochs), st_container=st.write()):
        model.fit(
            X_train,
            y_train,
            epochs=1,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_test, y_test),
        )

        # history model
        (train_loss, train_acc, val_loss, val_acc) = (
            model.history.history["loss"][0],
            model.history.history["accuracy"][0],
            model.history.history["val_loss"][0],
            model.history.history["val_accuracy"][0],
        )
        status_text.text(
            "Epoch: %d/%d - Training in progress... \ntrain loss: %.4f - train acc: %.3f \nval loss: %.4f - val acc: %.3f "
            % (epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc)
        )
        if train_loss < 0.005 or train_acc == 1.000:
            break

    # history model
    status_text.text(f"Epoch {epoch + 1}/{epochs} - Training completed!")
    st.write("train loss: %.4f - train acc: %.3f " % (train_loss, train_acc))
    st.write("val loss: %.4f - val acc: %.3f " % (val_loss, val_acc))

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    glob_y_test_pred_class = le.inverse_transform(y_pred)
    globy_y_test_class = le.inverse_transform(np.argmax(y_test, axis=1))
    # st.write(confusion_matrix(st.session_state.y_test, st.session_state.y_pred_class))

    glob_path_model = "models/teachable_machine_model_%s.h5" % (glob_str_kelas)
    model.save(glob_path_model)
    st.session_state.isModelTrained = 1
    # return True


def sidebar():
    CM_fig = ConfusionMatrixDisplay.from_predictions(
        globy_y_test_class, glob_y_test_pred_class
    )
    st.sidebar.markdown(
        "<h2 style='text-align:center;'>Confusion Matrix from Trainning</h2>",
        unsafe_allow_html=True,
    )
    st.sidebar.pyplot(CM_fig.figure_)

    # download model
    st.sidebar.markdown(
        "<h2 style='text-align:center;'>Download Model</h2>", unsafe_allow_html=True
    )
    st.sidebar.download_button(
        label="Download Model",
        data=open(glob_path_model, "rb").read(),
        file_name=glob_path_model,
    )


def get_ImagePredict():
    model = keras.models.load_model(glob_path_model)
    img = load_img(st.session_state.data_image_predict, target_size=(224, 224))
    X_test = np.array([img_to_array(img)]) / 255.0
    result = model.predict(X_test)
    return result


def show_result():
    st.markdown("<br><br><hr>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align:center;'> Prediksi Gambar </h1>",
        unsafe_allow_html=True,
    )

    radiopredict = st.radio(
        "Pilih Salah Satu",
        ("Upload Gambar", "Ambil Gambar dari Webcam"),
        key="radiopredict",
    )
    if radiopredict == "Upload Gambar":
        image_predict = st.file_uploader(
            "Upload Gambar",
            accept_multiple_files=False,
            key="data_image_predict",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
        )
    elif radiopredict == "Ambil Gambar dari Webcam":
        image_predict = st.camera_input(
            "Ambil Gambar dari Webcam",
            key="data_image_predict",
        )

    if image_predict:
        st.markdown("<h4>Image Predict</h4>", unsafe_allow_html=True)
        st.image(image_predict)
        st.markdown("<h4>Hasil</h4>", unsafe_allow_html=True)

        result = get_ImagePredict()
        y_pred = np.argmax(result, axis=1)
        y_pred_class = glob_input_kelas[y_pred[0]]

        st.write(
            "Gambar ini termasuk ke dalam kelas: %s - Probabilitas : %.3f"
            % (y_pred_class, result[0][y_pred] * 100)
        )
        for probability, kelas in zip(result[0], list(glob_input_kelas)):
            st.write("Kelas: %s - Probabilitas : %.3f" % (kelas, probability * 100))
            st.progress(int(probability * 100))
    else:
        st.info("Masukkan Gambar untuk prediksi", icon="ℹ️")


st.cache(suppress_st_warning=True)
