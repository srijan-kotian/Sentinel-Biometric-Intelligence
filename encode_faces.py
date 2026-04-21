import time
from imutils import paths
import face_recognition
import pickle
import cv2
import os
import argparse

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False, default="dataset", 
                help="path to input dataset directory")
ap.add_argument("-e", "--encodings-file", required=False, default="encodings.pickle", 
                help="path to serialized db of facial encodings")
ap.add_argument("-m", "--detection-method", type=str, default='hog', 
                help="face detection model to use: either 'hog' or 'cnn'")

def get_command_line_args():
    args = vars(ap.parse_args())
    return args['dataset'], args['encodings_file'], args['detection_method']

dataset, encodings_file, detection_method = get_command_line_args()

# Grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(dataset))

# Initialize the list of known encodings and known names
knownEncodings = []
knownNames = []
s = time.time()

# Loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # Extract the person name from the image path (folder name)
    name = imagePath.split(os.path.sep)[-2]
    print(f"[INFO] processing image [{name}] {i+1}/{len(imagePaths)}")

    # Load the input image
    image = cv2.imread(imagePath)
    if image is None:
        print(f"[WARNING] Could not load {imagePath}. Skipping.")
        continue

    # Convert from BGR (OpenCV) to RGB (dlib/face_recognition)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the (x,y)-coordinates of the bounding boxes
    boxes = face_recognition.face_locations(rgb_image, model=detection_method)

    # --- DATA SCIENCE VALIDATION ---
    # Ensure exactly one face is detected per sample to maintain data purity
    if len(boxes) != 1:
        print(f"[WARNING] Image {imagePath} has {len(boxes)} faces. Expected 1. Skipping.")
        continue

    # Compute the facial embedding (The 128-d vector/latent space representation)
    encodings = face_recognition.face_encodings(rgb_image, boxes)

    # Loop over the encodings (though we verified there's only 1)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

e = time.time()
print(f"[INFO] Encoding dataset took: {((e - s) / 60):.2f} minutes")

# Serialize (dump) the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}

with open(encodings_file, "wb") as f:
    f.write(pickle.dumps(data))

print(f"[SUCCESS] {len(knownEncodings)} embeddings saved to {encodings_file}")