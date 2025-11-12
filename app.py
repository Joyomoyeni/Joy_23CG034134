"""
Flask Web Application for Emotion Detection
Supports image upload and live camera capture
"""

from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import base64
from datetime import datetime
import sqlite3
import os
from werkzeug.utils import secure_filename
import io
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Configuration
USE_CUSTOM_MODEL = os.path.exists('emotion_detection_model_final.keras') or os.path.exists('emotion_detection_model_final.h5')

if USE_CUSTOM_MODEL:
    print("Loading custom trained model...")
    try:
        if os.path.exists('emotion_detection_model_final.keras'):
            CUSTOM_MODEL = tf.keras.models.load_model('emotion_detection_model_final.keras')
        else:
            CUSTOM_MODEL = tf.keras.models.load_model('emotion_detection_model_final.h5')
        print("Custom model loaded successfully!")
    except Exception as e:
        print(f"Error loading custom model: {e}")
        print("Falling back to DeepFace...")
        USE_CUSTOM_MODEL = False
        CUSTOM_MODEL = None
else:
    CUSTOM_MODEL = None
    print("No custom model found. Using DeepFace only.")

# Emotion labels and recommendations
EMOTION_LABELS = {
    'angry': 'Angry',
    'disgust': 'Disgust',
    'fear': 'Fear',
    'happy': 'Happy',
    'sad': 'Sad',
    'surprise': 'Surprise',
    'neutral': 'Neutral'
}

EMOTION_RECOMMENDATIONS = {
    'angry': {
        'message': 'Take a deep breath and try to relax.',
        'tips': [
            '🧘‍♂️ Practice deep breathing exercises',
            '🚶‍♀️ Go for a short walk to clear your mind',
            '🎵 Listen to calming music',
            '💬 Talk to someone you trust about your feelings',
            '✍️ Write down what\'s bothering you'
        ],
        'color': '#f8d7da'
    },
    'disgust': {
        'message': 'It\'s okay to feel uncomfortable sometimes.',
        'tips': [
            '🌬️ Get some fresh air',
            '🧼 Clean your surroundings for a fresh start',
            '🍵 Have a refreshing drink',
            '📱 Distract yourself with something pleasant',
            '🎨 Engage in a creative activity'
        ],
        'color': '#d1ecf1'
    },
    'fear': {
        'message': 'You\'re safe. Take it one step at a time.',
        'tips': [
            '🛡️ Remind yourself that you are safe',
            '💪 Focus on what you can control',
            '🤝 Reach out to a friend or family member',
            '📝 List your strengths and past victories',
            '🧘 Practice mindfulness or meditation'
        ],
        'color': '#e2e3e5'
    },
    'happy': {
        'message': 'That\'s wonderful! Keep spreading the positivity!',
        'tips': [
            '😊 Share your joy with others',
            '📸 Capture this moment with a photo',
            '🎉 Celebrate your achievements',
            '💝 Do something kind for someone else',
            '📓 Write about what made you happy today'
        ],
        'color': '#d4edda'
    },
    'sad': {
        'message': 'It\'s okay to feel sad. Be kind to yourself.',
        'tips': [
            '💙 Allow yourself to feel your emotions',
            '☎️ Call a friend or loved one',
            '🎬 Watch something uplifting',
            '🐾 Spend time with a pet if you have one',
            '🌟 Remember that this feeling is temporary'
        ],
        'color': '#cce5ff'
    },
    'surprise': {
        'message': 'Life is full of surprises! Embrace the moment.',
        'tips': [
            '🎊 Enjoy the unexpected moment',
            '📝 Journal about this surprise',
            '🤔 Reflect on what surprised you',
            '😄 Share the experience with others',
            '🌈 Stay open to new experiences'
        ],
        'color': '#fff3cd'
    },
    'neutral': {
        'message': 'You seem calm and balanced.',
        'tips': [
            '⚖️ Maintain your inner peace',
            '🎯 Use this clarity to plan your goals',
            '📚 This is a great time for focused work',
            '🧘‍♀️ Practice gratitude for the calm',
            '☕ Enjoy the present moment'
        ],
        'color': '#e7e7e7'
    }
}


def get_emotion_recommendation(emotion):
    """Get recommendation for detected emotion"""
    emotion_lower = emotion.lower()
    return EMOTION_RECOMMENDATIONS.get(emotion_lower, EMOTION_RECOMMENDATIONS['neutral'])


def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('emotion_detection.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotion_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT,
            image_data BLOB,
            detected_emotion TEXT NOT NULL,
            confidence REAL,
            all_emotions TEXT,
            source_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized successfully!")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_to_database(name, image_path, image_data, emotion_result, source_type):
    """Save emotion detection result to database"""
    try:
        conn = sqlite3.connect('emotion_detection.db')
        cursor = conn.cursor()

        # Extract emotion information
        dominant_emotion = emotion_result.get('dominant_emotion', 'Unknown')
        emotions = emotion_result.get('emotion', {})

        # Get confidence for dominant emotion
        confidence = emotions.get(dominant_emotion, 0.0) if emotions else 0.0

        # Convert all emotions to string
        all_emotions_str = str(emotions)

        cursor.execute('''
            INSERT INTO emotion_records 
            (name, image_path, image_data, detected_emotion, confidence, all_emotions, source_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, image_path, image_data, dominant_emotion, confidence, all_emotions_str, source_type))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Database error: {str(e)}")
        return False


def detect_emotion_custom_model(image_array):
    """
    Detect emotion using custom trained model

    Args:
        image_array: numpy array of the image

    Returns:
        dict: Detection results with emotions and confidence scores
    """
    try:
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array

        # Detect face using OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return {
                'success': False,
                'error': 'No face detected. Please ensure your face is clearly visible.'
            }

        # Get the first face
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]

        # Resize to 48x48 (FER2013 size)
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.reshape(1, 48, 48, 1) / 255.0

        # Predict
        predictions = CUSTOM_MODEL.predict(face_roi, verbose=0)[0]

        # Create emotion dictionary
        emotion_labels_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        emotion_dict = {}
        for i, label in enumerate(emotion_labels_list):
            emotion_dict[label] = float(predictions[i] * 100)

        # Get dominant emotion
        dominant_idx = np.argmax(predictions)
        dominant_emotion = emotion_labels_list[dominant_idx]

        return {
            'success': True,
            'dominant_emotion': dominant_emotion,
            'emotion': emotion_dict,
            'region': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
            'model_used': 'custom'
        }

    except Exception as e:
        return {
            'success': False,
            'error': f'Custom model error: {str(e)}'
        }


def detect_emotion_deepface(image_array):
    """
    Detect emotion using DeepFace with improved accuracy

    Args:
        image_array: numpy array of the image

    Returns:
        dict: Detection results with emotions and confidence scores
    """
    try:
        # Use opencv detector (no download needed!)
        result = DeepFace.analyze(
            img_path=image_array,
            actions=['emotion'],
            enforce_detection=True,
            detector_backend='opencv',  # Changed from retinaface
            silent=True
        )

        # Handle both single face and multiple faces
        if isinstance(result, list):
            result = result[0]

        return {
            'success': True,
            'dominant_emotion': result['dominant_emotion'],
            'emotion': result['emotion'],
            'region': result.get('region', {}),
            'model_used': 'deepface'
        }
    except ValueError as e:
        # Face not detected - try with less strict detection
        try:
            result = DeepFace.analyze(
                img_path=image_array,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            return {
                'success': True,
                'dominant_emotion': result['dominant_emotion'],
                'emotion': result['emotion'],
                'region': result.get('region', {}),
                'warning': 'Face detection uncertain - results may be less accurate',
                'model_used': 'deepface'
            }
        except Exception as e2:
            return {
                'success': False,
                'error': 'No face detected in the image. Please ensure your face is clearly visible and well-lit.'
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def detect_emotion(image_array):
    """
    Detect emotion using available models
    Tries custom model first, falls back to DeepFace
    """
    if USE_CUSTOM_MODEL:
        print("Using custom trained model...")
        result = detect_emotion_custom_model(image_array)
        if result['success']:
            return result
        print("Custom model failed, trying DeepFace...")

    return detect_emotion_deepface(image_array)


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/detect_from_upload', methods=['POST'])
def detect_from_upload():
    """Handle image upload and emotion detection"""
    try:
        # Get user name
        name = request.form.get('name', 'Anonymous')

        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})

        file = request.files['image']

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})

        if file and allowed_file(file.filename):
            # Read image
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)

            # Convert RGB to BGR for OpenCV compatibility
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            # Detect emotion
            result = detect_emotion(image_array)

            if not result['success']:
                return jsonify({'success': False, 'error': result['error']})

            # Save to database
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            # Save file
            with open(filepath, 'wb') as f:
                f.write(image_bytes)

            # Save to database
            save_to_database(
                name=name,
                image_path=filepath,
                image_data=image_bytes,
                emotion_result=result,
                source_type='upload'
            )

            # Prepare response with recommendations
            emotion_data = result['emotion']
            dominant_emotion = result['dominant_emotion']

            # Convert numpy float32 to regular Python float
            emotions_dict = {}
            for k, v in emotion_data.items():
                label = EMOTION_LABELS.get(k, k)
                emotions_dict[label] = float(round(float(v), 2))

            confidence = float(round(float(emotion_data[dominant_emotion]), 2))

            # Get recommendation
            recommendation = get_emotion_recommendation(dominant_emotion)

            response_data = {
                'success': True,
                'dominant_emotion': EMOTION_LABELS.get(dominant_emotion, dominant_emotion),
                'emotions': emotions_dict,
                'confidence': confidence,
                'recommendation': recommendation
            }

            # Add warning if present
            if 'warning' in result:
                response_data['warning'] = result['warning']

            return jsonify(response_data)

        return jsonify({'success': False, 'error': 'Invalid file type'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/detect_from_camera', methods=['POST'])
def detect_from_camera():
    """Handle live camera capture and emotion detection"""
    try:
        data = request.json
        name = data.get('name', 'Anonymous')
        image_data = data.get('image')

        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'})

        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)

        # Convert to numpy array
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)

        # Convert RGB to BGR
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Detect emotion
        result = detect_emotion(image_array)

        if not result['success']:
            return jsonify({'success': False, 'error': result['error']})

        # Save to database
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"camera_{timestamp}.jpg")

        # Save image
        with open(filepath, 'wb') as f:
            f.write(image_bytes)

        save_to_database(
            name=name,
            image_path=filepath,
            image_data=image_bytes,
            emotion_result=result,
            source_type='camera'
        )

        # Prepare response with recommendations
        emotion_data = result['emotion']
        dominant_emotion = result['dominant_emotion']

        # Convert numpy float32 to regular Python float
        emotions_dict = {}
        for k, v in emotion_data.items():
            label = EMOTION_LABELS.get(k, k)
            emotions_dict[label] = float(round(float(v), 2))

        confidence = float(round(float(emotion_data[dominant_emotion]), 2))

        # Get recommendation
        recommendation = get_emotion_recommendation(dominant_emotion)

        response_data = {
            'success': True,
            'dominant_emotion': EMOTION_LABELS.get(dominant_emotion, dominant_emotion),
            'emotions': emotions_dict,
            'confidence': confidence,
            'recommendation': recommendation
        }

        # Add warning if present
        if 'warning' in result:
            response_data['warning'] = result['warning']

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/get_records')
def get_records():
    """Retrieve all emotion detection records"""
    try:
        conn = sqlite3.connect('emotion_detection.db')
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, name, detected_emotion, confidence, source_type, timestamp
            FROM emotion_records
            ORDER BY timestamp DESC
            LIMIT 50
        ''')

        records = cursor.fetchall()
        conn.close()

        records_list = []
        for record in records:
            records_list.append({
                'id': record[0],
                'name': record[1],
                'emotion': record[2],
                'confidence': round(record[3], 2),
                'source': record[4],
                'timestamp': record[5]
            })

        return jsonify({'success': True, 'records': records_list})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


if __name__ == '__main__':
    # Initialize database on startup
    init_database()

    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)