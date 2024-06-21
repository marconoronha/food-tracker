import os
import webbrowser
import requests
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from werkzeug.utils import secure_filename
from threading import Timer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from all origins

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# URL do backend para classificação
BACKEND_URL = 'http://localhost:5001/predict'  # Alterar para o URL correto do seu backend

@app.route('/predict', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Enviar a imagem para o backend
            with open(file_path, 'rb') as f:
                response = requests.post(BACKEND_URL, files={'file': f})
            
            # Verificar se a resposta é bem-sucedida
            if response.status_code == 200:
                return jsonify(response.json()), 200
            else:
                return jsonify({'error': 'Failed to get prediction from backend'}), response.status_code

    except Exception as e:
        return jsonify({'error': f'Error during file upload: {str(e)}'}), 500

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

def open_browser():
    webbrowser.open_new('http://localhost:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()  # Open the browser after 1 second
    app.run(debug=True, host='0.0.0.0', port=5000)
