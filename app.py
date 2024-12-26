import streamlit as st
import functions as f

if "isModelTrained" not in st.session_state:
    st.session_state.isModelTrained = 0


def main():
    st.markdown(
        "<h1 style='text-align:center'>Teachable Machine</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align:center'>Image Classification</h3>",
        unsafe_allow_html=True,
    )

    # Form input numclass
    total_class_input = st.number_input(
        "Banyak Kelas", key="data_num_class", step=1, value=2
    )

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # Form input class and video duration
    if total_class_input > 1:
        try:
            for i in range(total_class_input):
                st.markdown(
                    "<h4 style='text-align:center'>Kelas {}</h4>".format(i + 1),
                    unsafe_allow_html=True,
                )
                col_kls, col_upload = st.columns(2)
                input_kelas = col_kls.text_input(
                    "Label {}".format(i + 1),
                    placeholder="Nama Kelas",
                    key="class_input{}".format(i),
                )
                input_image = col_upload.file_uploader(
                    "Input Gambar",
                    label_visibility="hidden",
                    accept_multiple_files=True,
                    key="image_input{}".format(i),
                    type=["jpg", "jpeg", "png"],
                )

                # with kol1:
                cb_record = st.checkbox(
                    "Record sample Camera?", key="accrecord_button{}".format(i)
                )

                col_samp, col_rcrd = st.columns(2, gap="large")
                if cb_record:
                    count_recordframe = col_samp.number_input(
                        "Banyak sample diambil",
                        value=100,
                        min_value=10,
                        step=10,
                        key="intrecord_button{}".format(i),
                    )

                    buttonrecord = col_rcrd.button(
                        "Record Frame Sample", key="record_button{}".format(i)
                    )

                    if buttonrecord:
                        if input_kelas:
                            recorded_frames = f.record_video(
                                input_kelas, count_recordframe
                            )
                            (
                                st.session_state["recorded_frames{}".format(i)],
                                st.session_state["recorded_class{}".format(i)],
                            ) = recorded_frames
                        else:
                            st.warning(
                                "Masukkan Nama kelas terlebih dahulu!", icon="⚠️"
                            )

                st.markdown("<br><hr><br>", unsafe_allow_html=True)

            # training model
            col1, col2, col3 = st.columns(3, gap="large")
            tuning_param = col2.checkbox("tuning parameter")
            btn_training = col2.button("Train Model", type="primary")

            if tuning_param:
                epochs_input = col2.number_input("Epochs", min_value=1, value=10)
                batch_size_input = col2.number_input("Batch Size", min_value=1, value=8)
            else:
                epochs_input = 10
                batch_size_input = 8
            if btn_training:
                f.trainingModel(epochs_input, batch_size_input)
        except:
            st.warning("Form Tidak boleh Kosong!", icon="⚠️")
    else:
        st.info("Minimal 2 Kelas", icon="ℹ️")

    if st.session_state.isModelTrained == 1:
        # print(st.session_state.isModelTrained)
        f.sidebar()
        f.show_result()
    else:
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='text-align:justify'>How to Use Teachable Machine?</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h4 style='text-align:justify'>1. Persiapan Data</h4>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Masukkan jumlah kelas dari data (minimal 2 kelas)</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Masukkan nama kelas</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Upload Masukkan gambar untuk setiap kelas atau dapat juga menggunakan fitur record sample mengggunakan webcam</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h4 style='text-align:justify'>2. Training</h4>",
            unsafe_allow_html=True,
        )
        # tuning parameter
        st.markdown(
            "<p style='text-align:justify'>- Centang Tuning Parameter untuk mengubah nilai Epochs dan Batch Size</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Tekan tombol Train Model</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Tunggu hingga proses training selesai</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Confusion matrix tampil pada sidebar sebagai hasil evaluasi</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Model dapat di download pada sidebar</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h4 style='text-align:justify'>3. Prediksi</h4>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Pilih gambar yang akan diprediksi atau dapat menggunakan fitur input gambar dari webcam</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Tekan tombol Predict</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Tunggu hingga proses prediksi selesai</p>",
            unsafe_allow_html=True,
        )
        # hasil prediksi
        st.markdown(
            "<p style='text-align:justify'>- Hasil prediksi dan kemungkinan kelas dari input akan muncul </p>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
