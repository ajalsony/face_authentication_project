import os
import cv2
from cv2 import face as cv2_face
import numpy as np
from PIL import Image
import flask
from flask import Flask, render_template, request, redirect, session, jsonify
import base64

class FaceAuthentication:
    def __init__(self):
        self.data_path = 'user_data'
        self.model_path = 'trained_model.yml'
        self.cascade_path = 'haarcascade_frontalface_default.xml'
        os.makedirs(self.data_path, exist_ok=True)

    def capture_samples(self, user_id, sample_count=50):
        """Capture face samples for training"""
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(self.cascade_path)
        
        # Clear previous samples
        existing_samples = [f for f in os.listdir(self.data_path) if f.startswith(f'face.{user_id}.')]
        for sample in existing_samples:
            os.remove(os.path.join(self.data_path, sample))

        count = 0
        while count < sample_count:
            ret, img = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                filename = os.path.join(self.data_path, f'face.{user_id}.{count}.jpg')
                cv2.imwrite(filename, face_img)
                count += 1

                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imshow('Face Capture', img)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

        cam.release()
        cv2.destroyAllWindows()
        return count > 0

    def train_model(self):
        """Train face recognition model"""
        recognizer = cv2_face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier(self.cascade_path)

        def get_images_and_labels():
            image_paths = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith('.jpg')]
            faces, ids = [], []

            for image_path in image_paths:
                pil_img = Image.open(image_path).convert('L')
                img_array = np.array(pil_img, 'uint8')
                user_id = int(os.path.splitext(os.path.basename(image_path))[0].split('.')[1])

                faces.append(img_array)
                ids.append(user_id)

            return faces, ids

        faces, ids = get_images_and_labels()
        recognizer.train(faces, np.array(ids))
        recognizer.write(self.model_path)
        print("Model trained successfully")

    def recognize_face(self, image_data):
        """Recognize face from base64 image"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None

            # Prepare image for recognition
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detector = cv2.CascadeClassifier(self.cascade_path)
            recognizer = cv2_face.LBPHFaceRecognizer_create()
            recognizer.read(self.model_path)

            faces = detector.detectMultiScale(gray, 1.2, 5)
            
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                id, confidence = recognizer.predict(face_img)
                
                if confidence < 100:
                    return id
            
            return None
        except Exception as e:
            print(f"Face recognition error: {e}")
            return None

# Flask Web Application
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
face_auth = FaceAuthentication()

# Simulated user database
USERS = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Generate user ID
        user_id = len(USERS) + 1
        USERS[user_id] = {
            'username': username,
            'password': password
        }
        
        # Capture face samples
        success = face_auth.capture_samples(user_id)
        
        if success:
            # Train model with new samples
            face_auth.train_model()
            return redirect('/login')
        else:
            return "Face registration failed"
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Check for face authentication
        if 'image' in request.form:
            image_data = request.form['image']
            user_id = face_auth.recognize_face(image_data)
            
            if user_id and user_id in USERS:
                session['user_id'] = user_id
                return jsonify({
                    'success': True, 
                    'username': USERS[user_id]['username']
                })
            
            return jsonify({'success': False})
        
        # Traditional login
        username = request.form['username']
        password = request.form['password']
        
        for user_id, user_info in USERS.items():
            if user_info['username'] == username and user_info['password'] == password:
                session['user_id'] = user_id
                return redirect('/dashboard')
        
        return "Login failed"
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/login')
    
    user = USERS.get(session['user_id'])
    return f"Welcome, {user['username']}!"

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)