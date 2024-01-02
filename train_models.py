import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
from speakerfeatures import extract_features
import warnings
import os

warnings.filterwarnings("ignore")

# path to training data
source = "C:\\Users\\DELL\\Desktop\\Mini-Project\\Speaker-identification-using-GMMs\\development_set"

# path where training speakers will be saved
dest = "C:\\Users\\DELL\\Desktop\\Mini-Project\\Speaker-identification-using-GMMs\\dest_models"

# Create the destination directory if it doesn't exist
if not os.path.exists(dest):
    os.makedirs(dest)

train_file = "C:\\Users\\DELL\\Desktop\\Mini-Project\\Speaker-identification-using-GMMs\\development_set_enroll.txt"

file_paths = open(train_file, 'r')

count = 1

# Extracting features for each speaker (5 files per speaker)
features = np.asarray(())
for path in file_paths:
    path = path.strip()
    print(path)

    # read the audio
    sr, audio = read(os.path.join(source, path))

    # extract 40-dimensional MFCC & delta MFCC features
    vector = extract_features(audio, sr)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    
    # when features of 5 files of the speaker are concatenated, then do model training
    if count == 5:
        gmm = GMM(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
        gmm.fit(features)

        # dumping the trained Gaussian model
        speaker_id = os.path.basename(path).split("-")[0]
        picklefile = speaker_id + ".gmm"
        with open(os.path.join(dest, picklefile), 'wb') as pickle_file:
            pickle.dump(gmm, pickle_file)
        print('+ modeling completed for speaker:', speaker_id, " with data point = ", features.shape)
        features = np.asarray(())
        count = 0
    
    count += 1
