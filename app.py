from flask import Flask, request, render_template
from pyngrok import ngrok
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import io
app = Flask(__name__)
# Load the model
model_path = '/content/drive/MyDrive/Plant_DD/Model/plant_disease_detector.h5'
model = load_model(model_path)
# Assuming train_generator is defined as in previous steps
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/Plant_DD/Model/Output_Datasets/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return 'No file uploaded.'
        file = request.files['file']
        if file.filename == '':
            return 'No file selected.'
        img = image.load_img(io.BytesIO(file.read()), target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction[0])
        class_name = list(train_generator.class_indices.keys())[class_idx]
        return class_name
    except Exception as e:
        return str(e), 500


# Run the app
if __name__ == '__main__':
    port = 5000
    public_url = ngrok.connect(port)
    print(f"Public URL: {public_url}")
    app.run(port=port)
