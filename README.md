# 🔍 Deepfake Detection System

A sophisticated AI-powered web application that detects deepfake videos using a hybrid CNN (MTCNN) + Vision Transformer model.

##  Features

- **High Accuracy**: ~92% accuracy on FaceForensics dataset
- **Hybrid AI Model**: Combines CNN feature extraction with Vision Transformer classification
- **Web Interface**: User-friendly drag-and-drop video upload
- **Real-time Analysis**: Frame-by-frame deepfake detection
- **Analytics Dashboard**: Track detection statistics and history
- **Face Detection**: MTCNN-based robust face extraction

##  Architecture

- **Backend**: Python Flask + PyTorch
- **Frontend**: HTML/CSS/JavaScript with Bootstrap
- **AI Model**: ConViTInspired (CNN + ViT hybrid)
- **Face Detection**: MTCNN
- **Dataset**: FaceForensics++

##  Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository**


2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Download pre-trained model**
   The trained model (`my_deepfake_model.pth`) is included via Git LFS. After cloning, the model should be automatically available.

### Training Your Own Model

1. **Open Jupyter Notebook**

```bash
jupyter notebook New_Paper.ipynb
```

2. **Run training cells (1-8)**
   - Downloads FaceForensics dataset via Kaggle
   - Processes video data and extracts faces
   - Trains the ConViTInspired model
   - Saves model as `my_deepfake_model.pth`

### Running the Web Application

```bash
python app.py
```

Navigate to `http://localhost:5000` to use the web interface.

##  Model Performance

- **Accuracy**: ~92% on test set
- **Architecture**: Hybrid CNN + Vision Transformer
- **Training Dataset**: FaceForensics++
- **Face Detection**: MTCNN with high precision
- **Input Size**: 224x224 RGB images
- **Classes**: Binary (Real/Fake)

## 🖥️ Usage

### Web Interface

1. **Upload Video**: Drag and drop video file on web interface
2. **Processing**: AI analyzes each frame for deepfake indicators
3. **Results**: View prediction confidence and frame-by-frame analysis
4. **Analytics**: Track detection history and statistics

### Programmatic Usage

```python
from your_model import ConViTInspired, predict_video

# Load trained model
model = ConViTInspired(num_classes=2)
model.load_state_dict(torch.load('my_deepfake_model.pth'))

# Predict on video
result = predict_video('path/to/video.mp4')
print(f"Prediction: {result}")
```

4. Wait for processing (may take a few minutes depending on video length)

### Viewing Results

The results page will show:

- **Overall Prediction**: Fake or Real with confidence percentage
- **Analysis Summary**: Number of frames analyzed, fake/real detections
- **Detected Faces**: Visual display of faces found in the video
- **Frame Analysis**: Detailed breakdown of each frame's prediction

### Analytics

The analytics section provides:

- Total videos processed
- Fake vs Real detection statistics
- Fake detection percentage
- Recent analysis history

## Technical Details

### Model Architecture

- **Base Model**: Vision Transformer (ViT) from Google
- **Custom Layers**: Convolutional layers for feature extraction
- **Face Detection**: MTCNN for face extraction from videos
- **Input Size**: 224x224 pixels

### Processing Pipeline

1. Video upload and validation
2. Face extraction from video frames (up to 10 frames)
3. Face preprocessing and resizing
4. Model inference on each detected face
5. Aggregation of results across all frames
6. Result storage and visualization

### File Structure

```
Resume_proj/
├── app.py                 # Flask backend application
├── templates/
│   └── index.html        # Frontend interface
├── requirements.txt      # Python dependencies
├── New_Paper.ipynb       # Model training notebook
├── my_deepfake_model.pth # Trained model (generated after training)
├── uploads/              # Temporary video storage (auto-created)
├── results/              # Analysis results storage (auto-created)
└── static/               # Static files (auto-created)
```

## System Requirements

- **Python**: 3.8 or higher
- **GPU**: CUDA-compatible GPU recommended for faster processing
- **RAM**: Minimum 8GB, 16GB recommended
- **Storage**: At least 5GB free space for model and temporary files

## Troubleshooting

### Model Not Loading

- Ensure `my_deepfake_model.pth` exists in the project directory
- Run the complete training notebook first

### Out of Memory Errors

- Reduce video size or length
- Close other applications to free up RAM
- Use CPU instead of GPU by modifying the device setting in `app.py`

### Slow Processing

- Processing time depends on video length and hardware
- GPU acceleration significantly improves speed
- Consider reducing the number of frames analyzed (modify `num_frames` parameter)

## API Endpoints

- `GET /`: Main interface
- `POST /upload`: Upload and analyze video
- `GET /analytics`: Get system analytics
- `GET /health`: Check system status

##  Project Structure

```
deepfake-detection/
├── app.py                 # Flask web application
├── New_Paper.ipynb        # Model training notebook
├── requirements.txt       # Python dependencies
├── my_deepfake_model.pth  # Trained model weights (Git LFS)
├── model_info.json        # Model metadata
├── templates/
│   └── index.html        # Web interface
├── static/               # CSS/JS assets
└── README.md            # This file
```

## 🔬 Technical Details

### Model Architecture

- **ConViTInspired**: Custom hybrid model
- **CNN Layers**: Feature extraction with channel expansion (3→16→3)
- **Vision Transformer**: Google's ViT-base-patch16-224 for classification
- **Input Processing**: MTCNN face detection + resize to 224x224
- **Output**: Binary classification with confidence scores

### Data Processing Pipeline

1. **Video Input**: MP4/AVI video files
2. **Face Detection**: MTCNN extracts faces from frames
3. **Preprocessing**: Resize to 224x224, normalize
4. **Model Inference**: CNN feature extraction + ViT classification
5. **Aggregation**: Average predictions across multiple frames

### Training Details

- **Optimizer**: Adam with learning rate 5e-5
- **Loss Function**: Weighted CrossEntropyLoss for class imbalance
- **Batch Size**: 8 (optimized for memory)
- **Epochs**: 10 with progress tracking
- **Data Split**: 80% training, 20% testing

##  Model File (Git LFS)

This project uses Git LFS to store the trained model file:

- **File**: `my_deepfake_model.pth` (~500MB)
- **Storage**: Git Large File Storage
- **Auto-download**: Happens automatically when you clone the repo

If you need to manually pull LFS files:

```bash
git lfs pull
```

##  Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request


##  Acknowledgments


- # Deepfake Face Extraction and Detection Using MTCNN-Vision Transformers-https://ieeexplore.ieee.org/document/10549578
- **FaceForensics++** dataset creators for providing high-quality deepfake data
- **Google Research** for the Vision Transformer architecture
- **PyTorch** and **Transformers** library maintainers
- **MTCNN** developers for robust face detection

## 🔧 Troubleshooting

### Common Issues

**Model file not found:**

```bash
# Ensure Git LFS is installed and pull large files
git lfs install
git lfs pull
```

**CUDA out of memory:**

```python
# Reduce batch size in training
batch_size = 4  # Instead of 8
```

**No faces detected:**

- Ensure video contains clear, front-facing faces
- Check video quality and lighting
- Try different frame sampling rates

## Security Notes

- Files are temporarily stored and automatically deleted after processing
- No user authentication implemented (suitable for local use only)
- For production deployment, add authentication and security measures


