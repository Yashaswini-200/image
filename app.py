from flask import Flask, request, jsonify 
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

from predictionFunction import predict_image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return "Welcome to the AI Image Classifier API!"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'up'}), 200

@app.route('/predict', methods=['POST'])
def handle_prediction():
    print("🔵 /predict endpoint hit")

    if 'file' in request.files:
        print("📁 File received in request")
        file = request.files['file']

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type.'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            result = predict_image(filepath)
        except Exception as e:
            return jsonify({'error': f'Failed to process the file: {str(e)}'}), 500
        finally:
            os.remove(filepath)

        return jsonify({'result': result})
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
