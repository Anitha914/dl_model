import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

# === Load your trained classifier model ===
classifier_model = load_model("oct_classifier_model.hdf5")

# === Load feature extractor ===
conv_base = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')

# === Class mapping ===
class_map = {0: "NORMAL", 1: "CNV", 2: "DME", 3: "DRUSEN"}

# === Caption templates ===
caption_templates = [
    "An OCT image showing signs of {}.",
    "This OCT scan indicates {}.",
    "Retinal features consistent with {} are visible.",
    "{} characteristics are observed in this OCT image.",
    "This is an OCT image of a patient with {}."
]

# === Function to extract features ===
def extract_feature(image_path):
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = conv_base.predict(image, verbose=0)
    return feature.flatten()

# === Function to generate caption ===
def generate_caption(image_path):
    feature = extract_feature(image_path)
    feature = np.expand_dims(feature, axis=0)
    
    # Predict class
    prediction = classifier_model.predict(feature)
    predicted_class = class_map[np.argmax(prediction)]
    
    # Pick a caption template (random or first)
    caption = caption_templates[0].format(predicted_class)
    
    return caption

# === Test with any image ===
test_image = r"D:\6th sem\7th sem\OCT_Project\OCT_small\train\CNV\CNV-53264-17.jpeg"  # change to your image
caption = generate_caption(test_image)
print("🖼️ Caption:", caption)
