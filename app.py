import streamlit as st
import numpy as np
import pickle
import sys
import types
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input as mobilenet_preprocess_input,
)
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input as vgg16_preprocess_input,
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM, Bidirectional, Embedding, Dense, Dropout


@tf.keras.utils.register_keras_serializable()
class CompatLSTM(LSTM):
    @classmethod
    def from_config(cls, config):
        config.pop("time_major", None)
        return super().from_config(config)


# Allow loading legacy pickle objects that reference keras.preprocessing.text
keras_preprocessing_text = types.ModuleType("keras.preprocessing.text")
keras_preprocessing_text.Tokenizer = Tokenizer
sys.modules.setdefault("keras.preprocessing.text", keras_preprocessing_text)

# Load your trained model (legacy H5 compatibility with newer Keras)
legacy_custom_objects = {
    "LSTM": CompatLSTM,
    "CompatLSTM": CompatLSTM,
    "Bidirectional": Bidirectional,
    "Embedding": Embedding,
    "Dense": Dense,
    "Dropout": Dropout,
}
model = None
model_load_error = None
try:
    model = tf.keras.models.load_model(
        'mymodel.h5',
        compile=False,
        custom_objects=legacy_custom_objects,
        safe_mode=False,
    )
except Exception as exc:
    model_load_error = exc

# Pick the feature extractor that matches the loaded caption model.
# VGG16 penultimate feature size: 4096
# MobileNetV2 penultimate feature size: 1280
feature_extractor = None
feature_preprocess = None
feature_extractor_name = ""
expected_feature_dim = None

if model is not None:
    expected_feature_dim = int(model.input_shape[0][-1])
    if expected_feature_dim == 4096:
        vgg16_model = VGG16(weights="imagenet")
        feature_extractor = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.layers[-2].output)
        feature_preprocess = vgg16_preprocess_input
        feature_extractor_name = "VGG16"
    else:
        mobilenet_model = MobileNetV2(weights="imagenet")
        feature_extractor = Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)
        feature_preprocess = mobilenet_preprocess_input
        feature_extractor_name = "MobileNetV2"

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
    
# Set custom web page title
st.set_page_config(page_title="Caption Generator App", page_icon="📷")

# Streamlit app
st.title("Image Caption Generator")
st.markdown(
    "Upload an image, and this app will generate a caption for it using a trained LSTM model."
)

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process uploaded image
if model_load_error is not None:
    st.error(f"Model failed to load in this environment: {model_load_error}")
    st.info("Use Python 3.8-3.10 with the original TensorFlow/Keras stack used to train this model.")
elif model is not None:
    st.caption(f"Feature extractor in use: {feature_extractor_name} (expected dim: {expected_feature_dim})")

if uploaded_image is not None and model is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Generated Caption")
    # Display loading spinner while processing
    with st.spinner("Generating caption..."):
        # Load image
        image = load_img(uploaded_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = feature_preprocess(image)

        # Extract features using the selected extractor
        image_features = feature_extractor.predict(image, verbose=0)

        # Max caption length
        max_caption_length = 34
        
        # Define function to get word from index
        def get_word_from_index(index, tokenizer):
            return next(
                (word for word, idx in tokenizer.word_index.items() if idx == index), None
        )

        # Generate caption using the model
        def predict_caption(model, image_features, tokenizer, max_caption_length):
            caption = "startseq"
            for _ in range(max_caption_length):
                sequence = tokenizer.texts_to_sequences([caption])[0]
                sequence = pad_sequences([sequence], maxlen=max_caption_length)
                yhat = model.predict([image_features, sequence], verbose=0)
                predicted_index = np.argmax(yhat)
                predicted_word = get_word_from_index(predicted_index, tokenizer)
                if predicted_word is None or predicted_word == "endseq":
                    break
                caption += " " + predicted_word
            return caption

        # Generate caption
        generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)

        # Remove startseq and endseq
        generated_caption = generated_caption.replace("startseq", "").replace("endseq", "")

    # Display the generated caption with custom styling
    st.markdown(
        f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
        f'<p style="font-style: italic;">“{generated_caption}”</p>'
        f'</div>',
        unsafe_allow_html=True
    )
