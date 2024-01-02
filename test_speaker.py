import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time

# Path to training data
source = "C:\\Users\\DELL\\Desktop\\Mini-Project\\Speaker-identification-using-GMMs\\development_set"
modelpath = "C:\\Users\\DELL\\Desktop\\Mini-Project\\Speaker-identification-using-GMMs\\dest_models"
test_file = "C:\\Users\\DELL\\Desktop\\Mini-Project\\Speaker-identification-using-GMMs\\development_set_test.txt"

file_paths = open(test_file, 'r')

gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

# Load the Gaussian gender Models
models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]

# Initialize accuracy counter
correct_identifications = 0
total_samples = 0

# Read the test directory and get the list of test audio files
for path in file_paths:
    path = path.strip()
    print(path)
    sr, audio = read(os.path.join(source, path))
    vector = extract_features(audio, sr)

    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        gmm = models[i]  # checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    detected_speaker = speakers[winner]
    
    # Get the ground truth label from the filename
    ground_truth_speaker = path.split("-")[0]

    print("\tDetected as - ", detected_speaker)

    # Update accuracy counters
    total_samples += 1
    if detected_speaker == ground_truth_speaker:
        correct_identifications += 1

    time.sleep(1.0)

# Calculate accuracy
accuracy = (correct_identifications / total_samples) * 100
print(f"\nAccuracy: {accuracy:.2f}%")
