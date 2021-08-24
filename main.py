import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow

tqdm.pandas()
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

target = 'Attractive'  # set this to Attractive or Babyface

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
pd.options.display.max_seq_items = 1000

cfd_df_raw = pd.read_csv("metadata.csv")  # TODO format file


# print(cfd_df_raw.head())


def getFileNames(model):
    files = []
    file_count = 0
    path = "Images/CFD/%s/" % model
    for r, d, f in os.walk(path):
        for file in f:  # BF-001 has several images
            if ('.jpg' in file) or ('.jpeg' in file) or '.png' in file:
                files.append(file)
    return files


cfd_df_raw["files"] = cfd_df_raw.Model.apply(getFileNames)
# print(cfd_df_raw[['Model', 'files']].head())

cfd_instances = []
for index, instance in cfd_df_raw.iterrows():
    folder = instance.Model
    score = instance[target]
    for file in instance.files:
        tmp_instance = []
        # tmp_instance.append((model, file, score))
        tmp_instance.append(folder)
        tmp_instance.append(file)
        tmp_instance.append(score)
        cfd_instances.append(tmp_instance)

df = pd.DataFrame(cfd_instances, columns=["folder", "file", "score"])
print(df[['file', 'score']].head())


def findEmotion(file):
    # file = CFD-WM-040-023-HO.jpg
    file_name = file.split(".")[0]  # [1] is jpg
    emotion = file_name.split("-")[4]
    return emotion


def findRace(file):
    # file = CFD-WM-040-023-HO.jpg
    file_name = file.split(".")[0]  # [1] is jpg
    race = file_name.split("-")[1][0]
    return race


def findGender(file):
    # file = CFD-WM-040-023-HO.jpg
    file_name = file.split(".")[0]  # [1] is jpg
    gender = file_name.split("-")[1][1]
    return gender


df['emotion'] = df.file.apply(findEmotion)
df['race'] = df.file.apply(findRace)
df['gender'] = df.file.apply(findGender)

# # include neutral, happen open mouth and happy close mouth
df = df[(df.emotion == 'N') | (df.emotion == 'HO') | (df.emotion == 'HC')]
df['file'] = "Images/CFD/" + df["folder"] + "/" + df['file']


def retrievePixels(path):
    img = image.load_img(path, grayscale=False, target_size=(224, 224))
    x = image.img_to_array(img).reshape(1, -1)[0]
    return x


#
df['pixels'] = df['file'].progress_apply(retrievePixels)

for index, instance in df[(df.race == 'W')  # A: Asian, B: Black, L: Latino, W: White]
                          & (df.gender == 'F')  # F: Female / M: Male
                          & (df.emotion == 'HO')  # HO: Happy Open Mouth, HC: Happy Closed Mouth, N: Neutral
].sort_values(by=['score'], ascending=False).head(3).iterrows():
    # for index, instance in df.sort_values(by=['score'], ascending = False).head(3).iterrows():
    img = instance.pixels
    img = img.reshape(224, 224, 3)
    img = img / 255

    plt.imshow(img)
    plt.show()
    print(instance.file)
    print("Attractiveness score: ", instance.score)

    print("-------------------")
