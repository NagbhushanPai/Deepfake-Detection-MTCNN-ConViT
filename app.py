# Flask Backend for Deepfake Detection
from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor
import cv2
from mtcnn import MTCNN
import os
import numpy as np
from PIL import Image
import json
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Model class (same as your notebook)
class ConViTInspired(nn.Module):
    def __init__(self, num_classes=2):
        super(ConViTInspired, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=num_classes, ignore_mismatched_sizes=True)
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.proj = nn.Conv2d(16, 3, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, pixel_values, labels=None):
        x = self.conv(pixel_values)
        x = self.relu(x)
        x = self.proj(x)
        outputs = self.vit(pixel_values=x, labels=labels)
        return outputs

# Global variables
model = None
image_processor = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load the trained model"""
    global model, image_processor
    try:
        model = ConViTInspired(num_classes=2).to(device)
        if os.path.exists('my_deepfake_model.pth'):
            model.load_state_dict(torch.load('my_deepfake_model.pth', map_location=device))
            model.eval()
            print("Model loaded successfully!")
        else:
            print("Warning: Model file not found. Please train the model first.")
        
        image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def extract_faces_from_video(video_path, num_frames=10):
    """Extract faces from video (same as your notebook)"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    face_images = []  # Store face images for visualization
    detector = MTCNN()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // num_frames)

    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(frame_rgb)
        if faces:
            x, y, w, h = faces[0]['box']
            face = frame_rgb[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224, 224))
            frames.append(Image.fromarray(face_resized))
            
            # Store original face for visualization
            face_images.append({
                'frame_number': i,
                'face_array': face,
                'bbox': faces[0]['box']
            })
    cap.release()
    return frames, face_images

def predict_video_detailed(video_path):
    # Predict video with detailed results
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        frames, face_images = extract_faces_from_video(video_path, num_frames=10)
        if not frames:
            return {"error": "No faces detected in video"}
        
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for i, frame in enumerate(frames):
                inputs = image_processor(frame, return_tensors='pt').pixel_values.to(device)
                outputs = model(inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1).item()
                confidence = torch.max(probabilities).item()
                
                predictions.append(pred)
                confidences.append(confidence)
        
        # Calculate overall prediction
        avg_pred = np.mean(predictions) > 0.5
        avg_confidence = np.mean(confidences)
        
        # Convert face images to base64 for frontend
        face_images_b64 = []
        for i, face_data in enumerate(face_images[:5]):  # Limit to 5 faces for display
            img = Image.fromarray(face_data['face_array'])
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            face_images_b64.append({
                'image': img_str,
                'frame_number': face_data['frame_number'],
                'prediction': 'Fake' if predictions[i] else 'Real',
                'confidence': f"{confidences[i]*100:.1f}%"
            })
        
        result = {
            'overall_prediction': 'Fake' if avg_pred else 'Real',
            'confidence': f"{avg_confidence*100:.1f}%",
            'frame_predictions': [{'frame': i, 'prediction': 'Fake' if p else 'Real', 'confidence': f"{c*100:.1f}%"} 
                                for i, (p, c) in enumerate(zip(predictions, confidences))],
            'face_images': face_images_b64,
            'summary': {
                'total_frames_analyzed': len(predictions),
                'fake_count': sum(predictions),
                'real_count': len(predictions) - sum(predictions),
                'avg_confidence': f"{avg_confidence*100:.1f}%"
            }
        }
        
        return result
    except Exception as e:
        return {"error": f"Error processing video: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Secure filename and save
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Process video
        result = predict_video_detailed(file_path)
        
        # Save result
        result_id = str(uuid.uuid4())
        result_data = {
            'id': result_id,
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'result': result
        }
        
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
        with open(result_path, 'w') as f:
            json.dump(result_data, f)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify({'result_id': result_id, 'result': result})

@app.route('/result/<result_id>')
def get_result(result_id):
    try:
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        else:
            return jsonify({'error': 'Result not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analytics')
def analytics():
    try:
        results = []
        for filename in os.listdir(app.config['RESULTS_FOLDER']):
            if filename.endswith('.json'):
                with open(os.path.join(app.config['RESULTS_FOLDER'], filename), 'r') as f:
                    data = json.load(f)
                    results.append(data)
        
        # Calculate analytics
        total_videos = len(results)
        fake_videos = sum(1 for r in results if r['result'].get('overall_prediction') == 'Fake')
        real_videos = total_videos - fake_videos
        
        analytics_data = {
            'total_videos_processed': total_videos,
            'fake_videos_detected': fake_videos,
            'real_videos_detected': real_videos,
            'fake_percentage': (fake_videos / total_videos * 100) if total_videos > 0 else 0,
            'recent_results': sorted(results, key=lambda x: x['timestamp'], reverse=True)[:10]
        }
        
        return jsonify(analytics_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

if __name__ == '__main__':
    print("Loading model...")
    model_loaded = load_model()
    if model_loaded:
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please check if 'my_deepfake_model.pth' exists.")
