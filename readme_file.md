# Emotion Detection Web Application

A comprehensive emotion detection system using deep learning with DeepFace (VGGFace/ResNet50) and the FER2013 dataset.

## 📋 Features

- **Image Upload Detection**: Upload images to detect emotions
- **Live Camera Detection**: Real-time emotion detection from webcam
- **Database Storage**: All detections stored in SQLite database
- **Modern UI**: Beautiful, responsive web interface
- **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (for live detection)
- Internet connection (for downloading pre-trained models)

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download FER2013 Dataset** (Optional - for training):
   - Go to [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
   - Download and extract to project root
   - Place `fer2013.csv` in the project directory

### Running the Application

1. **Start the Flask server**:
```bash
python app.py
```

2. **Open your browser** and navigate to:
```
http://localhost:5000
```

3. **Use the application**:
   - Enter your name
   - Either upload an image OR use live camera
   - View emotion detection results
   - Check recent detections in the records table

## 📁 Project Structure

```
STUDENTS-SURNAME_MAT.NO_EMOTION_DETECTION_WEB_APP/
│
├── app.py                          # Flask backend application
├── model.py                        # Model training script
├── requirements.txt                # Python dependencies
├── link_to_my_web_app.txt         # Hosting link
├── emotion_detection.db           # SQLite database (auto-created)
│
├── templates/
│   └── index.html                 # Web interface
│
└── uploads/                       # Uploaded images (auto-created)
```

## 🎓 Training Your Own Model (Optional)

If you want to train the model from scratch:

1. **Download FER2013 dataset** from Kaggle

2. **Run training script**:
```bash
python model.py
```

3. **Training outputs**:
   - `best_emotion_model.h5` - Best model during training
   - `best_emotion_model_finetuned.h5` - Fine-tuned model
   - `emotion_detection_model_final.h5` - Final trained model
   - `training_history.png` - Training metrics visualization

**Note**: The app uses DeepFace's pre-trained models by default, so training is optional.

## 🌐 Deployment

### Option 1: Render (Recommended)

1. Create account at [render.com](https://render.com)
2. Create new Web Service
3. Connect your GitHub repository
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Environment**: Python 3

### Option 2: Heroku

1. Create account at [heroku.com](https://heroku.com)
2. Install Heroku CLI
3. Deploy:
```bash
heroku login
heroku create your-app-name
git push heroku main
```

### Option 3: PythonAnywhere

1. Create account at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload files
3. Create web app with Flask
4. Configure WSGI file

### Option 4: Railway

1. Create account at [railway.app](https://railway.app)
2. Connect GitHub repository
3. Deploy automatically

## 📊 Database Schema

The SQLite database stores:
- User name
- Image path and data
- Detected emotion
- Confidence score
- All emotion probabilities
- Source type (upload/camera)
- Timestamp

## 🔧 Configuration

### Changing Models

In `app.py`, modify the `detect_emotion_deepface` function:
```python
result = DeepFace.analyze(
    img_path=image_array,
    actions=['emotion'],
    enforce_detection=False,
    detector_backend='opencv'  # Options: opencv, ssd, dlib, mtcnn, retinaface
)
```

### Adjusting File Size Limit

In `app.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

## 🐛 Troubleshooting

### Camera Not Working
- Ensure browser permissions are granted
- Use HTTPS in production (required for camera access)
- Try different browsers

### Model Download Issues
- Ensure stable internet connection
- DeepFace downloads models on first use
- Models cached in `~/.deepface/weights/`

### Database Errors
- Delete `emotion_detection.db` to reset
- Check file permissions
- Ensure SQLite is installed

## 📝 API Endpoints

- `GET /` - Main web interface
- `POST /detect_from_upload` - Upload image for detection
- `POST /detect_from_camera` - Live camera detection
- `GET /get_records` - Retrieve detection history
- `GET /health` - Health check

## 🤖 Technical Details

### Models Used
- **DeepFace Framework**: Meta's facial analysis library
- **Backend Options**: VGGFace, ResNet50, Facenet
- **Emotion Analysis**: 7-class emotion classification

### Training Details (model.py)
- **Base Model**: ResNet50 with ImageNet weights
- **Input Size**: 224x224x3
- **Data Augmentation**: Rotation, shift, flip, zoom
- **Callbacks**: Early stopping, learning rate reduction
- **Fine-tuning**: Last 20 layers unfrozen

## 📚 Resources

- [DeepFace Documentation](https://github.com/serengil/deepface)
- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

## 📧 Submission

1. **GitHub Repository**:
   - Create repository named: `SURNAME_MAT.NO_EMOTION_DETECTION`
   - Push all files
   - Ensure README is complete

2. **Hosting**:
   - Deploy to free platform
   - Add link to `link_to_my_web_app.txt`

3. **Zip and Submit**:
```bash
zip -r SURNAME_MAT.NO_EMOTION_DETECTION_WEB_APP.zip .
```
   - Send to: odunayo.osofuye@covenantuniversity.edu.ng

## ⚠️ Important Notes

- Replace `SURNAME` and `MAT.NO` with your actual details
- Test thoroughly before submission
- Include all required files
- Ensure hosting link works
- Database should persist data

## 🎯 Grading Criteria

- ✅ Code functionality
- ✅ Model accuracy
- ✅ Web interface quality
- ✅ Database implementation
- ✅ Deployment success
- ✅ Code documentation

## 📄 License

This project is for educational purposes.

---

**Good luck with your submission! 🚀**
