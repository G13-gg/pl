import streamlit as st
import tempfile
import librosa
from tensorflow.keras.models import load_model
import numpy as np

# Fungsi untuk memuat model
@st.cache(allow_output_mutation=True)
def load_deep_learning_model(model_path):
    model = load_model(model_path)
    return model

# Fungsi untuk memproses file audio MP3
def process_audio_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file.seek(0)
        audio_data, sample_rate = librosa.load(temp_file.name, sr=None)  # Memuat audio dengan sampling rate asli

    return audio_data, sample_rate  # Mengembalikan data audio dan sampling rate

# Fungsi untuk melakukan prediksi dengan model
def predict_with_model(model, audio_data):
    # Lakukan prediksi dengan model
    # Misalnya, kita akan memprediksi label dari audio
    # Di sini, `audio_data` harus diproses sesuai dengan input model
    # Contoh sederhana: Prediksi dummy
    prediction = np.random.uniform(0, 1, size=(1, 7))  # Contoh hasil prediksi
    return prediction

def main():
    st.title('Aplikasi Evaluasi Bacaan Al-Fatihah')
    st.write('Selamat datang di aplikasi evaluasi bacaan Al-Fatihah.')

    # Section untuk upload file audio
    st.header('Upload Audio Surah Al-Fatihah')
    uploaded_file = st.file_uploader("Pilih file audio MP3", type="mp3")

    if uploaded_file is not None:
        # Proses audio yang diunggah
        audio_data, sample_rate = process_audio_file(uploaded_file)

        # Menampilkan audio yang diunggah (jika perlu)
        # st.audio(uploaded_file, format='audio/mp3', sample_rate=sample_rate)

        # Button untuk melakukan prediksi
        if st.button('Prediksi'):
            # Load model
            model_path = 'D:\Kuliah\S6\Studi Independen\project akhir\Bacaan Quran\Sahabat_Quran.keras'  
            model = load_deep_learning_model(model_path)

            # Lakukan prediksi dengan model
            prediction = predict_with_model(model, audio_data)
            st.write('Hasil Prediksi:')
            st.write(prediction)  # Menampilkan hasil prediksi

if __name__ == '__main__':
    main()
