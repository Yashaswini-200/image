from flask import Flask, request, jsonify 
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import logging

# Import actual prediction functions
from predictionFunction import predict_image, predict_from_url

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to the AI Image Classifier API!"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'up'}), 200

# 🔧 Upload folder setup
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB limit

# ✅ Allowed image formats
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 🚀 Predict route
@app.route('/predict', methods=['POST'])
def handle_prediction():
    print("🔵 /predict endpoint hit")

    # 🖼 File upload
    if 'file' in request.files:
        print("📁 File received in request")
        file = request.files['file']

        if not allowed_file(file.filename):
            print("❌ Invalid file type")
            return jsonify({'error': 'Invalid file type. Only .png, .jpg, .jpeg files are allowed!'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"📂 Saving file to: {filepath}")
        file.save(filepath)

        try:
            result = predict_image(filepath)
            print(f"✅ Prediction result: {result}")
        except Exception as e:
            print(f"🔥 Error during file prediction: {str(e)}")
            return jsonify({'error': f'Failed to process the file: {str(e)}'}), 500
        finally:
            os.remove(filepath)
            print(f"🧹 Deleted uploaded file: {filepath}")

        return jsonify({'result': result})

    # 🌐 Image URL
    if request.is_json and request.json and 'url' in request.json:
        image_url = request.json['url']
        print(f"🌐 URL received: {image_url}")

        if not image_url:
            print("❌ URL is empty")
            return jsonify({'error': 'URL is required for this request'}), 400

        try:
            result = predict_from_url(image_url)
            print(f"✅ Prediction result from URL: {result}")
        except Exception as e:
            print(f"🔥 Error during URL prediction: {str(e)}")
            return jsonify({'error': f'Failed to process the image from URL: {str(e)}'}), 500

        return jsonify({'result': result})

    print("❌ No file or URL provided")
    return jsonify({'error': 'No file or URL provided'}), 400

# 🏁 Run the app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5500, debug=True)