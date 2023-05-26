import tensorflow as tf
from flask import Flask, request, jsonify
from flask import render_template
from flask_cors import CORS
import os
import numpy as np
from keras.models import load_model
from pathlib import Path
import moviepy.editor as moviepy
import subprocess


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
@app.route('/')
def index():
    return render_template('index.html')

VALID_SPLIT = 0.3

# Seed to use when shuffling the dataset and the noise
SHUFFLE_SEED = 43
BATCH_SIZE = 128
SCALE = 0.5
SAMPLING_RATE = 16000
DATASET_ROOT = "A:\\AI\\Reconnaissancee Voix\\justone"

# The folders in which we will put the audio samples and the noise samples
AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"

DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    model = load_model("model.h5")
    noise_paths = []
    for subdir in os.listdir(DATASET_NOISE_PATH):
        subdir_path = Path(DATASET_NOISE_PATH) / subdir
        if os.path.isdir(subdir_path):
            noise_paths += [
                os.path.join(subdir_path, filepath)
                for filepath in os.listdir(subdir_path)
                if filepath.endswith(".wav")
            ]

    noises = []
    for path in noise_paths:
        sample = load_noise_sample(path)
        if sample:
            noises.extend(sample)
    noises = tf.stack(noises)

    

    class_names = os.listdir(DATASET_AUDIO_PATH)
    audio_paths = []
    labels = []
    for label, name in enumerate(class_names):
        print(
            "Processing speaker(name) {}".format(
                name,
            )
            
        )
        print(
            "Processing speaker(label) {}".format(
                label,
            )
            
        )
        dir_path = Path(DATASET_AUDIO_PATH) / name
        speaker_sample_paths = [
            os.path.join(dir_path, filepath)
            for filepath in os.listdir(dir_path)
            if filepath.endswith(".wav")
        ]
        audio_paths += speaker_sample_paths
        labels += [label] * len(speaker_sample_paths)
        
    num_val_samples = int(VALID_SPLIT * len(audio_paths))
    valid_audio_paths = audio_paths[-num_val_samples:]
    valid_labels = labels[-num_val_samples:]
    SAMPLES_TO_DISPLAY = 1

    test_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
    test_ds = test_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
        BATCH_SIZE
    )

    test_ds = test_ds.map(lambda x, y: (add_noise(x, noises, scale=SCALE), y))
    predictions={}
    for audios, labels in test_ds.take(1):
        # Get the signal FFT
        ffts = audio_to_fft(audios)
        # Predict
        y_pred = model.predict(ffts)
        # Take random samples
        rnd = np.random.randint(0, len(audios), SAMPLES_TO_DISPLAY)
        audios = audios.numpy()[rnd, :, :]
        labels = labels.numpy()[rnd]
        y_pred = np.argmax(y_pred, axis=-1)[rnd]


        print(
                "Speaker:\33{} {}\33[0m\tPredicted:\33{} {}\33[0m".format(
                    "[92m" if labels[0] == y_pred[0] else "[91m",
                    class_names[labels[0]],
                    "[92m" if labels[0] == y_pred[0] else "[91m",
                    class_names[y_pred[0]],
                )
            )
        predictions = {int(labels[0]): int(y_pred[0])}

        
    
    return predictions


def load_noise_sample(path):
    sample, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1
    )
    if sampling_rate == SAMPLING_RATE:
        # Number of slices of 16000 each that can be generated from the noise sample
        slices = int(sample.shape[0] / SAMPLING_RATE)
        sample = tf.split(sample[: slices * SAMPLING_RATE], slices)
        return sample
    else:
        print("Sampling rate for {} is incorrect. Ignoring it".format(path))
        return None


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


def add_noise(audio, noises=None, scale=0.5):
    if noises is not None:
        # Create a random tensor of the same size as audio ranging from
        # 0 to the number of noise stream samples that we have.
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

        # Get the amplitude proportion between the audio and the noise
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        # Adding the rescaled noise to audio
        audio = audio + noise * prop * scale
    return audio



@app.route('/save_audio', methods=['GET','POST'])
def save_audio():
    audio = request.files['audio']
    print("Received file content type:", audio.content_type)
    audio.save('A:\\AI\\Reconnaissancee Voix\\justone\\audio\\Killian Boisseau\\record.webm')  
    input_file = r"A:\\AI\\Reconnaissancee Voix\\justone\\audio\\Killian Boisseau\\record.webm"
    output_file = r"A:\\AI\\Reconnaissancee Voix\\justone\\audio\\Killian Boisseau\\record.wav"
    command = ['ffmpeg','-y', '-i', input_file, output_file]
    subprocess.run(command, shell=True)
    os.remove(input_file)
    prediction = predict()
    return prediction

if __name__ == "__main__":
    app.run()