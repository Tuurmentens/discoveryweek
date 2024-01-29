import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Function to preprocess the image
def preprocess_image(image_path):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Open the image and resize it to be at least 224x224
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Normalize the image
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    return data

# Streamlit app
st.title("Image Classification with Keras")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Make predictions when an image is uploaded
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image and make predictions
    input_data = preprocess_image(uploaded_file)
    predictions = model.predict(input_data)

    # Display the top prediction
    top_prediction_index = np.argmax(predictions[0])
    top_prediction_label = class_names[top_prediction_index]

    st.write(f"Prediction: {top_prediction_label}")
    st.write(f"Confidence: {predictions[0][top_prediction_index]:.2%}")

    # Turn the image into a numpy array
    image_array = np.asarray(Image.open(uploaded_file).convert("RGB"))

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:])
    print("Confidence Score:", confidence_score)
