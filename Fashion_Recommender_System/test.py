import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

# loading the feature list and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))


model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# taking a new image and extracting features and passing it to the model.
img = image.load_img('sample/jersey.jpeg',target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# Finding the top-k nearest images using KNN
# Metric used = Euclidean Distance
# We can use cosine similarity as well.

neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
#neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='cosine')
# give the input data to .fit function
neighbors.fit(feature_list)

# finding the nearest k neighbours for the normalized result.
distances,indices = neighbors.kneighbors([normalized_result])

# printing the indices of the vectors which are nearest to the given image.
print(indices)

for file in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file])
    # to display the image.
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    # to make the screen wait.
    cv2.waitKey(0)