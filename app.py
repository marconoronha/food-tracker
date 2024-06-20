import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import requests

app = Flask(__name__)

# Caminho para o modelo treinado
MODEL_PATH = './model/resnet18_fine_tuned.pth'
CLASSES_PATH = './meta/classes.txt'

# Configurações da API Edamam
EDAMAM_APP_ID = 'da62227c'  # Substitua pelo seu ID do aplicativo Edamam
EDAMAM_APP_KEY = 'a1e80e7c8ca34d27af3abdf02e55166c'  # Substitua pela sua chave do aplicativo Edamam
EDAMAM_URL = 'https://api.edamam.com/api/food-database/v2/parser'

# Carregar a lista de classes
def load_classes(classes_path):
    with open(classes_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

class_names = load_classes(CLASSES_PATH)

# Carregar o modelo
def load_model(model_path):
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model(MODEL_PATH)

# Transformação para pré-processar a imagem
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Função para prever a classe da imagem
def predict(image_path, model):
    image = Image.open(image_path).convert('RGB')  # Converter para RGB
    image = transform(image).unsqueeze(0)  # Adiciona uma dimensão de batch
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    return preds.item()

# Função para obter informações nutricionais da API Edamam
def get_nutritional_info(food_item):
    # Substituir underscores por espaços
    food_item = food_item.replace('_', ' ')
    
    params = {
        'app_id': EDAMAM_APP_ID,
        'app_key': EDAMAM_APP_KEY,
        'ingr': food_item
    }
    response = requests.get(EDAMAM_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'parsed' in data and len(data['parsed']) > 0:
            food_info = data['parsed'][0]['food']
            return food_info
        elif 'hints' in data and len(data['hints']) > 0:
            food_info = data['hints'][0]['food']
            return food_info
        else:
            return {'error': 'No nutritional information found for the given food item'}
    else:
        return {'error': 'Failed to fetch nutritional information'}

# Endpoint para fazer upload da imagem e classificá-la
@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        # Predizer a classe da imagem
        class_id = predict(file_path, model)
        
        # Mapear o id da classe para o nome da classe
        class_name = class_names[class_id]
        
        # Obter informações nutricionais da API Edamam
        nutritional_info = get_nutritional_info(class_name)
        
        return jsonify({'class_id': class_id, 'predicted_class_name': class_name, 'nutritional_info': nutritional_info})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
