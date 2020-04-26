import os
from flask import Flask, request, render_template, jsonify
from keras.models import load_model
from tensorflow import keras
import numpy as np
import pandas as pd
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
# load model
Xception_model = load_model('models/Xception_model.h5')
# summarize model.
Xception_model.summary()
dog_names = pd.read_csv('dog_names.csv').values.tolist()

def extract_Xception(tensor):
	from keras.applications.xception import Xception, preprocess_input
	return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))
from keras.preprocessing import image


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def Xception_predict_breed(img_path):
    """
     Extract bottleneck features and return predicted dog breed

    Parameters:
      image path : Path for the image for breed prediction
    Returns:
        predicted dog breed

    """
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    predicted_vector = Xception_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]


def predict_breed(img_path):
    result = Xception_predict_breed(img_path)
    print(result)
    return result[1].split('.')[1]

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload_file():
    data = {"success": False, "dog": ''}
    if request.files.get('image'):
        file = request.files['image']
        filename = file.filename
        filepath = os.path.join('img', filename)
        file.save(filepath)

        prediction = predict_breed(file)
        data = {"success": True, "dog": prediction}
    return jsonify(data)

if __name__ == '__main__':
    app.run(threaded=False)
