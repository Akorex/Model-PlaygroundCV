"""An extension of the code for predicting the images fed to a neural network. The original code
can be found in the original project's directory. This one has been reworked to fit the purpose
of using in a Flask webapp
"""

# import dependency relating to TensorFlow model prediction
import os
 # for feeding image to neural network
from keras.preprocessing.image import load_img, img_to_array
from keras import models
import pickle
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings


# for the pets
def predict_pets(file_path):
    # some parameters
    image_height = 150
    image_width = 150

    # preprocess the image
    image = load_img(file_path, target_size=(image_height, image_width))

    # load the model
    model = models.load_model(r'app/main/model_artifacts/pets-model')

    # convert image to array & clip to 0-1 range
    image = img_to_array(image)
    image = np.array(image)
    image = image[:]/255
    image = np.expand_dims(image, axis=0)

    # feed image to model & predict
    predict_proba = model.predict([image])

     # translate for use
    if predict_proba < 0.5:
        return "a cat."
    else:
        return "a dog."

