from flask import Flask, request, render_template, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the trained model
model = load_model('Model/rice_classifier.h5')

# Preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            img = preprocess_image(file_path)
            prediction = model.predict(img)
            result = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
            return render_template('index.html', result=result, image_path=file_path)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
