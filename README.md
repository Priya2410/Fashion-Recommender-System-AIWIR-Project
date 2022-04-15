# Fashion-Recommender-System-AIWIR-Project

### Dataset Used : https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

### https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

### Technologies Used : Deeplearning - CNN , Tranfer Learning ( ResNet50)

### We directly make use of ResNet50

# Overall Approach :
- Import the model [ CNN Model - ResNet ]
- It has been trained already on a dataset called - ImageNet
- The model is present in Keras.
- This ResNet model is used for Feature Extraction.
- The dataset has 44,000 images
- The image uploaded by the user -> feature extraction -> check with dataset and recommend.
- Model based approach of recommendation
- The features -> 2048 for each image
- Once the features are extracted -> export the features
- Provide the recommendations
- We'll use these features -> the feature vector and compare it with the feature vector of the new image and compare them using KNN and get the nearest neighbours and recommend it.
- Collaborative Filtering





