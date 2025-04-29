import os
import cv2
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import face_recognition
import speech_recognition as sr
from pymongo import MongoClient
from picamera2 import Picamera2
from scipy.spatial.distance import cosine
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from time import time, sleep
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
from sklearn.decomposition import PCA
import RPi.GPIO as GPIO
import pygame


# === PIR and GPIO Setup ===
PIR_PIN = 17
MOTOR_PIN = 23       # Motor relay control (lock/unlock)

# === Motor & Locking Setup ===
REED_SWITCH_PIN = 26    # GPIO pin for reed switch (input)
MOTOR_IN1 = 27          # Motor direction pin 1
MOTOR_IN2 = 22          # Motor direction pin 2

# === LED Setup ===
RED_LED_PIN = 5
ORANGE_LED_PIN = 6
GREEN_LED_PIN = 13
YELLOW_LED_PIN = 19

# === Setup GPIO ===
GPIO.setmode(GPIO.BCM)

# Reed Switch Setup
GPIO.setup(REED_SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Main Lock Motor Setup
GPIO.setup(MOTOR_IN1, GPIO.OUT)
GPIO.setup(MOTOR_IN2, GPIO.OUT)

# LEDs Setup
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(ORANGE_LED_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.setup(YELLOW_LED_PIN, GPIO.OUT)

# PIR Sensor Setup
GPIO.setup(PIR_PIN, GPIO.IN)

# Ensure LEDs and motor are initially off
GPIO.output(GREEN_LED_PIN, GPIO.LOW)
GPIO.output(RED_LED_PIN, GPIO.LOW)
GPIO.output(MOTOR_IN1, GPIO.LOW) 
GPIO.output(MOTOR_IN2, GPIO.LOW)

pygame.mixer.init()
                

# === Device and DB Info ===
device_serial_number = "LKT005"
mongo_uri = "mongodb+srv://LocktAdmin:L0cktForever@locktcluster.dfui7.mongodb.net/?retryWrites=true&w=majority&appName=LocktCluster"
flask_base_url = "https://locktsolutions.com"

client = MongoClient(mongo_uri)
db = client.LocktDatabase
users_collection = db.users

known_face_encodings = []
known_face_names = []
voice_profiles = {}
eigenface_projections = []
eigenface_names = []
pca_model = None

recognizer = sr.Recognizer()

def play_sound(filename):
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            sleep(0.1)
    except Exception as e:
            print(f"Error playing sound {filename}: {e}")
            

def extract_mfcc(file_path):
    try:
        fs, audio = wav.read(file_path)
        if len(audio) < 400:
            print(f"Audio too short: {len(audio)} samples")
            return None
        features = mfcc(audio, samplerate=fs, nfft=2048)
        return np.mean(features, axis=0)
    except Exception as e:
        print(f"MFCC extraction error: {e}")
        return None

def train_eigenfaces(face_images):
    face_vectors = [img.flatten() for img in face_images]
    face_vectors = np.array(face_vectors)
    n_samples = face_vectors.shape[0]
    n_components = min(10, n_samples)
    pca = PCA(n_components=n_components)
    pca.fit(face_vectors)
    return pca

def project_to_eigenfaces(pca, face_image):
    face_vector = face_image.flatten().reshape(1, -1)
    return pca.transform(face_vector)

def load_face_images():
    face_images = []
    for sub_user in user.get('sub_users', []):
        name = sub_user['name']
        for photo_filename in sub_user.get('photos', []):
            try:
                url = f"{flask_base_url}/static/uploads/{photo_filename}"
                response = requests.get(url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
                image_np = np.array(image)

                face_locations = face_recognition.face_locations(image_np)
                face_encs = face_recognition.face_encodings(image_np, face_locations)
                if face_encs:
                    known_face_encodings.append(face_encs[0])
                else:
                    print(f"No face encoding found in {photo_filename} for {name}")

                image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                image_gray = cv2.resize(image_gray, (200, 200))

                face_images.append(image_gray)
                known_face_names.append(name)
                print(f"Loaded face for {name}")
            except Exception as e:
                print(f"Failed to load image {photo_filename} for {name}: {e}")
    return face_images

print(f"Looking for user with device serial number: {device_serial_number}")
user = users_collection.find_one({'serial_number': device_serial_number})
if not user:
    print("No user found with that serial number.")
    GPIO.cleanup()
    exit()

face_images = load_face_images()
pca_model = train_eigenfaces(face_images)

eigenface_projections = []
eigenface_names = []
for img, name in zip(face_images, known_face_names):
    projection = project_to_eigenfaces(pca_model, img)
    eigenface_projections.append(projection)
    eigenface_names.append(name)

for sub_user in user.get('sub_users', []):
    name = sub_user['name']
    voice_filename = sub_user.get('voice_sample')
    if voice_filename:
        try:
            voice_url = f"{flask_base_url}/static/voice/{voice_filename}?v={int(time())}"
            print(f"Downloading voice: {voice_url}")
            response = requests.get(voice_url)
            response.raise_for_status()
            mp3_path = f"{name}.mp3"
            wav_path = f"{name}.wav"
            with open(mp3_path, "wb") as f:
                f.write(response.content)
            audio = AudioSegment.from_file(mp3_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(wav_path, format="wav")
            mfcc_profile = extract_mfcc(wav_path)
            if mfcc_profile is not None:
                voice_profiles[name] = mfcc_profile
                print(f"Stored MFCC for {name}")
            else:
                print(f"Failed to extract MFCC for {name}")
        except Exception as e:
            print(f"Error loading voice for {name}: {e}")

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888"}))
picam2.start()

print("System ready. Waiting for motion to begin recognition...")

try:
    while True:
        if GPIO.input(PIR_PIN):
            print("Motion detected. Starting authentication process...")
            play_sound("/home/raspberry/BeginningFacialRecognition.wav")

            face_attempts = 0
            authenticated = False

            while face_attempts < 3 and not authenticated:
                print(f"Attempt {face_attempts + 1}: Show your face to the camera.")
                frame = picam2.capture_array()
                small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                cv2.imshow("Face Recognition", frame)

                face_locations = face_recognition.face_locations(rgb_small)
                face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

                for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                    top *= 2; right *= 2; bottom *= 2; left *= 2
                    matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.45)
                    name = "Unknown"

                    if True in matches:
                        index = matches.index(True)
                        name = known_face_names[index]
                        print(f"Face recognized as {name}. Checking with Eigenfaces...")

                        live_gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
                        live_gray = cv2.resize(live_gray, (200, 200))
                        live_projection = project_to_eigenfaces(pca_model, live_gray)

                        best_match, best_dist = None, float('inf')
                        for proj, proj_name in zip(eigenface_projections, eigenface_names):
                            dist = np.linalg.norm(proj - live_projection)
                            if dist < best_dist:
                                best_dist, best_match = dist, proj_name

                        if best_match == name:
                            print(f"Eigenfaces match. Proceeding with voice verification...")
                            GPIO.output(YELLOW_LED_PIN, GPIO.HIGH)  #turns on yellow LED to signal user for voice recognition
                            play_sound("/home/raspberry/BeginVocalRecognition.wav")
                            voice_attempts = 0

                            while voice_attempts < 3:
                                with sr.Microphone() as source:
                                    try:
                                        print("Recording voice...")
                                        audio = recognizer.listen(source, timeout=5)
                                        with NamedTemporaryFile(delete=False, suffix=".mp3") as raw_mp3:
                                            raw_mp3.write(audio.get_wav_data())
                                            raw_mp3_path = raw_mp3.name
                                        audio_segment = AudioSegment.from_file(raw_mp3_path)
                                        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
                                        audio_segment.export("temp_voice.wav", format="wav")

                                        live_mfcc = extract_mfcc("temp_voice.wav")
                                        stored_mfcc = voice_profiles.get(name)
                                        if stored_mfcc is not None and live_mfcc is not None:
                                            similarity = 1 - cosine(stored_mfcc, live_mfcc)
                                            print(f"Voice similarity score: {similarity:.2f}")
                                            if similarity > 0.7:
                                                print(f"{name} authenticated.")
                                                authenticated = True
                                                break
                                            else:
                                                print("Voice mismatch.")
                                                play_sound("/home/raspberry/NotAMatch.wav")
                                        else:
                                            print("Missing or invalid MFCC.")
                                    except Exception as e:
                                        print("Voice error:", e)
                                voice_attempts += 1
                                
                        GPIO.output(YELLOW_LED_PIN, GPIO.LOW) #turns off yellow led

                face_attempts += 1
                if not authenticated:
                    print("Retrying in 3 seconds...")
                    play_sound("/home/raspberry/NotAMatch.wav")
                    sleep(3)
                else:
                    break

            cv2.destroyAllWindows()

            if authenticated:
                print("Access granted")
                GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
                GPIO.output(RED_LED_PIN, GPIO.LOW)
                play_sound("/home/raspberry/UnlockingDoor.wav")
                GPIO.output(MOTOR_IN1, GPIO.HIGH)  # Turn on motor to unlock
                GPIO.output(MOTOR_IN2, GPIO.LOW)

                sleep(5)  # Keep lock open for 5 seconds
                GPIO.output(MOTOR_IN1, GPIO.LOW)  # lock again
                GPIO.output(MOTOR_IN2, GPIO.LOW)
                GPIO.output(GREEN_LED_PIN, GPIO.LOW)
            else:
                print("Authentication failed. Waiting for next motion event...")
                play_sound("/home/raspberry/NotAMatch.wav")
                GPIO.output(RED_LED_PIN, GPIO.HIGH)
                sleep(2)
                GPIO.output(RED_LED_PIN, GPIO.LOW)
        else:
            sleep(1)  # Polling interval for PIR sensor

except KeyboardInterrupt:
    print("Exiting program...")
    GPIO.cleanup()
