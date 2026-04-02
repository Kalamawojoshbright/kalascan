from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
import io
import json
import os
import requests
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

app = Flask(__name__)

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70

# ── Model download from Google Drive ─────────────────────────
MODEL_LINKS = {
    'maize':  os.environ.get('MAIZE_MODEL_URL'),
    'beans':  os.environ.get('BEANS_MODEL_URL'),
    'banana': os.environ.get('BANANA_MODEL_URL'),
    'tomato': os.environ.get('TOMATO_MODEL_URL'),
    'coffee': os.environ.get('COFFEE_MODEL_URL'),
}

CLASS_DATA = {
    'maize':  ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy'],
    'beans':  ['angular_leaf_spot', 'bean_rust', 'healthy'],
    'banana': ['Cordana', 'Healthy', 'Panama Disease', 'Yellow and Black Sigatoka'],
    'tomato': ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
               'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy'],
    'coffee': ['Healthy', 'Miner', 'Phoma', 'Rust'],
}

def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None
    )
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def download_model(crop, url):
    path = f'/tmp/{crop}_model.keras'
    if os.path.exists(path):
        return path
    print(f'Downloading {crop} model...')
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f'{crop} model downloaded!')
    return path

# Load all models on startup
crop_models = {}
print('Loading models...')
for crop, url in MODEL_LINKS.items():
    if url:
        try:
            path = download_model(crop, url)
            classes = CLASS_DATA[crop]
            model = build_model(len(classes))
            model.load_weights(path)
            crop_models[crop] = model
            print(f'✅ {crop} loaded')
        except Exception as e:
            print(f'❌ {crop} failed: {e}')
print(f'Ready! Loaded: {list(crop_models.keys())}')

# ── Disease database ──────────────────────────────────────────
disease_db = {
    'Blight': {
        'severity': 'High',
        'description': 'Blight causes large brown lesions on maize leaves, reducing photosynthesis.',
        'treatments': [
            'Remove and destroy infected leaves immediately',
            'Apply copper-based fungicide every 7-10 days',
            'Avoid overhead watering — water at the base only',
            'Rotate crops to a different field next season',
            'Ensure good air circulation between plants'
        ]
    },
    'Common_Rust': {
        'severity': 'Medium',
        'description': 'Common Rust appears as orange-brown pustules on both sides of leaves.',
        'treatments': [
            'Apply fungicide containing mancozeb or chlorothalonil',
            'Plant rust-resistant maize varieties next season',
            'Remove heavily infected leaves from the field',
            'Avoid planting in the same field repeatedly',
            'Monitor crop early and treat at first signs'
        ]
    },
    'Gray_Leaf_Spot': {
        'severity': 'High',
        'description': 'Gray Leaf Spot causes rectangular gray-brown lesions between leaf veins.',
        'treatments': [
            'Apply strobilurin-based fungicide early in the season',
            'Use resistant hybrid maize varieties',
            'Reduce crop residue by tilling after harvest',
            'Ensure proper plant spacing for good air flow',
            'Avoid planting maize after maize in the same field'
        ]
    },
    'angular_leaf_spot': {
        'severity': 'High',
        'description': 'Angular Leaf Spot causes brown angular spots on bean leaves, reducing yield.',
        'treatments': [
            'Apply mancozeb or copper-based fungicide',
            'Use certified disease-free seeds next season',
            'Remove and burn infected plant debris',
            'Avoid working in field when plants are wet',
            'Space plants properly to reduce humidity'
        ]
    },
    'bean_rust': {
        'severity': 'Medium',
        'description': 'Bean Rust causes reddish-brown pustules on leaves, weakening the plant.',
        'treatments': [
            'Apply triazole or strobilurin fungicide',
            'Remove heavily infected leaves immediately',
            'Plant resistant bean varieties',
            'Avoid overhead irrigation',
            'Crop rotation with non-legume crops'
        ]
    },
    'Cordana': {
        'severity': 'Medium',
        'description': 'Cordana Leaf Spot causes oval brown spots with yellow margins on banana leaves.',
        'treatments': [
            'Apply copper-based fungicide monthly',
            'Remove and destroy heavily infected leaves',
            'Improve drainage around banana plants',
            'Avoid wounding plants during farm work',
            'Keep field clean of dead plant material'
        ]
    },
    'Panama Disease': {
        'severity': 'Critical',
        'description': 'Panama Disease is a deadly fungal wilt that can destroy entire banana plantations.',
        'treatments': [
            'Remove and destroy infected plants immediately',
            'Do NOT replant banana in the same soil',
            'Disinfect all farming tools after use',
            'Plant resistant varieties like Cavendish',
            'Report to local agricultural extension office'
        ]
    },
    'Yellow and Black Sigatoka': {
        'severity': 'High',
        'description': 'Sigatoka causes yellow and black streaks on leaves, reducing fruit quality.',
        'treatments': [
            'Apply systemic fungicide every 3 weeks',
            'Remove infected leaves and burn them',
            'Ensure adequate spacing between plants',
            'Use mulch to reduce spore splashing',
            'Apply potassium fertilizer to strengthen plants'
        ]
    },
    'Tomato___Bacterial_spot': {
        'severity': 'High',
        'description': 'Bacterial Spot causes dark water-soaked spots on tomato leaves and fruit.',
        'treatments': [
            'Apply copper-based bactericide spray',
            'Remove and destroy infected plant parts',
            'Avoid overhead watering',
            'Use disease-free certified seeds',
            'Rotate tomato crops every 2-3 years'
        ]
    },
    'Tomato___Early_blight': {
        'severity': 'Medium',
        'description': 'Early Blight causes dark brown spots with concentric rings on older leaves.',
        'treatments': [
            'Apply mancozeb or chlorothalonil fungicide',
            'Remove infected lower leaves immediately',
            'Mulch around plants to prevent spore splash',
            'Water at base, not on leaves',
            'Stake plants for better air circulation'
        ]
    },
    'Tomato___Late_blight': {
        'severity': 'Critical',
        'description': 'Late Blight is a fast-spreading disease that can destroy an entire crop in days.',
        'treatments': [
            'Apply fungicide immediately — do not wait',
            'Remove all infected plants and burn them',
            'Do not compost infected material',
            'Apply preventive fungicide to healthy plants nearby',
            'Avoid working in field when wet to stop spreading'
        ]
    },
    'Tomato___Leaf_Mold': {
        'severity': 'Medium',
        'description': 'Leaf Mold causes yellow patches on upper leaf surface with mold underneath.',
        'treatments': [
            'Improve ventilation in greenhouse or field',
            'Apply fungicide containing chlorothalonil',
            'Reduce humidity by spacing plants further apart',
            'Remove infected leaves promptly',
            'Avoid wetting foliage when watering'
        ]
    },
    'Tomato___Septoria_leaf_spot': {
        'severity': 'Medium',
        'description': 'Septoria Leaf Spot causes small circular spots with dark borders on leaves.',
        'treatments': [
            'Apply mancozeb fungicide at first sign',
            'Remove infected leaves from lower plant',
            'Mulch soil to prevent spore splash',
            'Avoid overhead watering',
            'Practice 2-year crop rotation'
        ]
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'severity': 'Medium',
        'description': 'Spider Mites cause yellowing and speckling on leaves with fine webbing visible.',
        'treatments': [
            'Spray plants with strong water jets to dislodge mites',
            'Apply neem oil or insecticidal soap spray',
            'Introduce natural predators like ladybugs',
            'Keep plants well watered — mites prefer dry conditions',
            'Apply miticide if infestation is severe'
        ]
    },
    'Tomato___Target_Spot': {
        'severity': 'Medium',
        'description': 'Target Spot causes circular brown lesions with concentric rings on leaves.',
        'treatments': [
            'Apply fungicide containing azoxystrobin',
            'Remove infected leaves and dispose safely',
            'Avoid dense planting to improve air flow',
            'Water early morning so leaves dry during day',
            'Monitor regularly and treat at first signs'
        ]
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'severity': 'Critical',
        'description': 'Yellow Leaf Curl Virus causes yellowing and curling — spread by whiteflies.',
        'treatments': [
            'Remove and destroy infected plants immediately',
            'Control whitefly population with insecticide',
            'Use yellow sticky traps to catch whiteflies',
            'Plant virus-resistant tomato varieties',
            'Use reflective mulch to repel whiteflies'
        ]
    },
    'Tomato___Tomato_mosaic_virus': {
        'severity': 'High',
        'description': 'Mosaic Virus causes mottled yellow-green patterns on leaves.',
        'treatments': [
            'Remove and destroy infected plants',
            'Wash hands thoroughly before handling plants',
            'Disinfect all tools with bleach solution',
            'Control aphids that spread the virus',
            'Plant resistant varieties next season'
        ]
    },
    'Miner': {
        'severity': 'Medium',
        'description': 'Coffee Leaf Miner causes white blotch patterns on leaves from larvae tunneling.',
        'treatments': [
            'Apply systemic insecticide to affected plants',
            'Remove and destroy heavily mined leaves',
            'Introduce natural parasitic wasps as biocontrol',
            'Avoid excessive nitrogen fertilizer',
            'Monitor plants weekly during dry season'
        ]
    },
    'Phoma': {
        'severity': 'High',
        'description': 'Phoma causes brown circular lesions on coffee leaves leading to defoliation.',
        'treatments': [
            'Apply copper-based fungicide immediately',
            'Remove infected leaves and branches',
            'Improve drainage around coffee plants',
            'Avoid overhead irrigation',
            'Prune plants for better air circulation'
        ]
    },
    'Rust': {
        'severity': 'Critical',
        'description': 'Coffee Leaf Rust is the most destructive coffee disease — orange powder on leaves.',
        'treatments': [
            'Apply copper fungicide or triazole immediately',
            'Remove heavily infected leaves and burn them',
            'Plant rust-resistant coffee varieties',
            'Ensure proper spacing between plants',
            'Report severe outbreaks to agricultural office'
        ]
    },
    'Healthy': {
        'severity': 'None',
        'description': 'Your plant looks healthy! Keep up the good farming practices.',
        'treatments': [
            'Continue regular watering schedule',
            'Apply balanced fertilizer as needed',
            'Monitor regularly for early signs of disease',
            'Keep weeds away from the crop',
            'Maintain good drainage in the field'
        ]
    },
    'healthy': {
        'severity': 'None',
        'description': 'Your plant looks healthy! Keep up the good farming practices.',
        'treatments': [
            'Continue regular watering schedule',
            'Apply balanced fertilizer as needed',
            'Monitor regularly for early signs of disease',
            'Keep weeds away from the crop',
            'Maintain good drainage in the field'
        ]
    },
}

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image', 'message': 'Please take a photo first.'}), 400

    crop = request.form.get('crop', 'maize').lower()

    if crop not in crop_models:
        return jsonify({'error': 'Model not ready', 'message': f'{crop.capitalize()} model is not loaded yet.'}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = crop_models[crop]
    classes = CLASS_DATA[crop]

    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index])
    predicted_class = classes[predicted_index]

    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({
            'error': 'Image not recognised',
            'message': f'This does not look like a {crop} leaf. Please take a clearer photo inside the box.'
        })

    info = disease_db.get(predicted_class, {
        'severity': 'Unknown',
        'description': f'Detected: {predicted_class}',
        'treatments': ['Please consult your local agricultural extension officer.']
    })

    return jsonify({
        'disease': predicted_class,
        'confidence': round(confidence * 100, 1),
        'severity': info['severity'],
        'description': info['description'],
        'treatments': info['treatments']
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
