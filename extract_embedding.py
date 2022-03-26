# face verification with the VGGFace2 model
from fileinput import filename
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from utils import preprocess_input
from vggface import VGGFace
from tqdm import tqdm
import pandas as pd
import os
import re
import csv
import face_recognition

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    img = plt.imread(filename)

    detector = MTCNN()
    try:
        results = detector.detect_faces(img)
        x1, y1, width, height = results[0]['box']
    except:
        image = face_recognition.load_image_file(filename)
        results = face_recognition.face_locations(image)
        results = [*results[0]]
        x1, y1, width, height = results

    x2, y2 = x1 + width, y1 + height

    # extract the face
    face = img[y1:y2, x1:x2]

    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
    # extract faces
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = np.asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)

    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    pred = model.predict(samples)
    return pred


# get embeddings of all available photos and save to .csv file
def csv_save(list_info):
    with open("photos_db.csv", "a", encoding='UTF-8') as f:
        writer = csv.writer(f, delimiter=";", lineterminator="\n")

        if open("photos_db.csv", "r").read() == "":
            writer.writerow(["filename", "ps_name", "embedding"])
        
        writer.writerow(list_info) 

p_exist = list(pd.read_csv('photos_db.csv', sep=';').iloc[:,0])

photos_dir = f'{os.getcwd()}/ps_photos'
photos_lst = [os.path.join(path, name) for path, subdirs, files in os.walk(photos_dir) for name in files]
photos_lst = [x for x in photos_lst if os.path.basename(x) not in p_exist]
photos_lst = [x for x in photos_lst if any(ext in x.lower() for ext in ['.png', '.jpeg', '.jpg'])]


# add embeddings of the photo that not in db
for photo in tqdm(photos_lst, position=0, leave=True):
    try:
        basefile = os.path.basename(photo)
        try:
            csv_save([basefile, re.match(r'(.*)_.*', os.path.splitext(basefile)[0]).group(1), str(get_embeddings([photo]).tolist())])
        except:
            csv_save([basefile, os.path.splitext(basefile)[0], str(get_embeddings([photo]).tolist())])
    except Exception as e:
        print(f"Can not detect a face on image - {basefile}")
        pass

# delete photos with no face detection
p_exist = list(pd.read_csv('photos_db.csv', sep=';').iloc[:,0])
photos_lst = [x for x in photos_lst if os.path.basename(x) not in p_exist]

for i in photos_lst:
    os.remove(i)
