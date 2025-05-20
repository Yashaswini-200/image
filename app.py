from flask import Flask, request, jsonify 
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

from predictionFunction import predict_image  # Your prediction function

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB max upload

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

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Save the uploaded file
    file.save(filepath)
    print(f"📁 File saved at {filepath}")

    try:
        # Pass ONLY filepath, no model argument here
        result = predict_image(filepath)
        print(f"🔍 Prediction result: {result}")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({'error': f'Failed to process the file: {str(e)}'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"🧹 Removed uploaded file {filepath}")

    return jsonify({'result': result})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)