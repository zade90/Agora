"""
Enhanced voice recognition system with real audio processing capabilities.
This module handles actual voice recording, processing, and user identification.
"""
import os
import json
import pickle
import numpy as np
import librosa
import soundfile as sf
import speech_recognition as sr
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import webrtcvad
from datetime import datetime
import uuid
import threading
import time

class RealVoiceRecognitionSystem:
    """Enhanced voice recognition with real audio processing."""
    
    def __init__(self, voice_profiles_dir, model_file):
        self.voice_profiles_dir = voice_profiles_dir
        self.model_file = model_file
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.voice_model = None
        self.scaler = StandardScaler()
        self.voice_profiles = {}
        self.vad = webrtcvad.Vad(2)  # Voice Activity Detection
        
        # Create directories
        os.makedirs(voice_profiles_dir, exist_ok=True)
        
        # Load existing model and profiles
        self.load_voice_model()
        self.load_voice_profiles()
        
        # Calibrate microphone
        self.calibrate_microphone()
    
    def calibrate_microphone(self):
        """Calibrate microphone for ambient noise."""
        try:
            with self.microphone as source:
                print("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Microphone calibrated successfully.")
        except Exception as e:
            print(f"Error calibrating microphone: {e}")
    
    def extract_voice_features(self, audio_data, sr=16000):
        """Extract comprehensive voice features for recognition."""
        try:
            # Ensure audio is the right format
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Extract MFCC features (most important for voice recognition)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            mfcc_delta = np.mean(librosa.feature.delta(mfccs), axis=1)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
            
            # Prosodic features
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            # Formant-like features
            stft = librosa.stft(audio_data)
            spectral_contrast = librosa.feature.spectral_contrast(S=np.abs(stft), sr=sr)
            
            # Combine all features
            features = np.concatenate([
                mfcc_mean,                                    # 13 features
                mfcc_std,                                     # 13 features  
                mfcc_delta,                                   # 13 features
                [np.mean(spectral_centroids), np.std(spectral_centroids)],  # 2 features
                [np.mean(spectral_rolloff), np.std(spectral_rolloff)],      # 2 features
                [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],  # 2 features
                [np.mean(zero_crossing_rate), np.std(zero_crossing_rate)],  # 2 features
                [pitch_mean, pitch_std],                      # 2 features
                np.mean(spectral_contrast, axis=1)            # 7 features
            ])
            
            return features
        except Exception as e:
            print(f"Error extracting voice features: {e}")
            return None
    
    def record_audio_sample(self, duration=3):
        """Record audio sample from microphone."""
        try:
            with self.microphone as source:
                print(f"Recording for {duration} seconds...")
                # Record audio
                audio = self.recognizer.listen(source, timeout=duration+1, phrase_time_limit=duration)
                
                # Convert to numpy array
                audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize
                
                return audio_data, audio
            
        except sr.WaitTimeoutError:
            print("No speech detected in the given time.")
            return None, None
        except Exception as e:
            print(f"Error recording audio: {e}")
            return None, None
    
    def register_voice_profile(self, user_name, care_type, num_samples=5):
        """Register a new voice profile with multiple samples."""
        try:
            print(f"Starting voice registration for {user_name}")
            print("Please speak clearly in a quiet environment.")
            
            user_id = str(uuid.uuid4())
            voice_samples = []
            audio_files = []
            
            sample_phrases = [
                "Hello, this is my voice registration sample one",
                "I am setting up my voice profile for care assistance",
                "Please help me with my daily health and wellness routine",
                "I would like to schedule reminders and manage my activities",
                "Thank you for providing personalized care support"
            ]
            
            for i in range(num_samples):
                print(f"\nSample {i+1}/{num_samples}")
                print(f"Please say: '{sample_phrases[i]}'")
                input("Press Enter when ready to record...")
                
                audio_data, audio_obj = self.record_audio_sample(duration=4)
                
                if audio_data is not None:
                    # Extract features
                    features = self.extract_voice_features(audio_data)
                    if features is not None:
                        voice_samples.append(features)
                        
                        # Save audio file
                        audio_filename = f"{user_id}_sample_{i+1}.wav"
                        audio_path = os.path.join(self.voice_profiles_dir, audio_filename)
                        
                        # Save as WAV file
                        sf.write(audio_path, audio_data, 16000)
                        audio_files.append(audio_filename)
                        
                        print(f"✓ Sample {i+1} recorded successfully")
                    else:
                        print(f"✗ Failed to process sample {i+1}")
                        return False, "Failed to process voice sample"
                else:
                    print(f"✗ Failed to record sample {i+1}")
                    return False, "Failed to record voice sample"
            
            if len(voice_samples) < 3:
                return False, "Need at least 3 successful voice samples"
            
            # Create voice profile
            profile = {
                'user_id': user_id,
                'name': user_name,
                'care_type': care_type,
                'features': voice_samples,
                'audio_files': audio_files,
                'created_at': datetime.now().isoformat(),
                'sample_count': len(voice_samples)
            }
            
            # Save profile
            profile_file = os.path.join(self.voice_profiles_dir, f"{user_id}_profile.pkl")
            with open(profile_file, 'wb') as f:
                pickle.dump(profile, f)
            
            self.voice_profiles[user_id] = profile
            
            # Retrain model
            success = self.train_voice_model()
            if success:
                print(f"✓ Voice profile registered successfully for {user_name}")
                return True, user_id
            else:
                return False, "Failed to train voice recognition model"
                
        except Exception as e:
            return False, f"Registration error: {e}"
    
    def train_voice_model(self):
        """Train the voice recognition model with all profiles."""
        try:
            if len(self.voice_profiles) < 2:
                print("Need at least 2 users to train voice recognition model")
                return False
            
            X = []
            y = []
            
            # Collect all features
            for user_id, profile in self.voice_profiles.items():
                for features in profile['features']:
                    X.append(features)
                    y.append(user_id)
            
            X = np.array(X)
            
            # Handle NaN or infinite values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split for validation
            if len(X) > 10:  # Only if we have enough samples
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, y_train = X_scaled, y
            
            # Train SVM with optimized parameters
            self.voice_model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
            
            self.voice_model.fit(X_train, y_train)
            
            # Save model
            model_data = {
                'model': self.voice_model,
                'scaler': self.scaler,
                'user_mapping': {uid: profile['name'] for uid, profile in self.voice_profiles.items()},
                'feature_dim': X.shape[1]
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Validation accuracy if possible
            if len(X) > 10:
                accuracy = self.voice_model.score(X_test, y_test)
                print(f"Voice recognition model trained with accuracy: {accuracy:.2f}")
            else:
                print("Voice recognition model trained successfully")
            
            return True
            
        except Exception as e:
            print(f"Error training voice model: {e}")
            return False
    
    def load_voice_model(self):
        """Load trained voice recognition model."""
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.voice_model = model_data['model']
                    self.scaler = model_data['scaler']
                print("Voice recognition model loaded successfully")
        except Exception as e:
            print(f"Error loading voice model: {e}")
    
    def load_voice_profiles(self):
        """Load all voice profiles from directory."""
        try:
            for filename in os.listdir(self.voice_profiles_dir):
                if filename.endswith('_profile.pkl'):
                    user_id = filename.replace('_profile.pkl', '')
                    profile_path = os.path.join(self.voice_profiles_dir, filename)
                    
                    with open(profile_path, 'rb') as f:
                        profile = pickle.load(f)
                        self.voice_profiles[user_id] = profile
            
            print(f"Loaded {len(self.voice_profiles)} voice profiles")
        except Exception as e:
            print(f"Error loading voice profiles: {e}")
    
    def identify_speaker_from_audio(self, audio_data):
        """Identify speaker from audio data."""
        try:
            if self.voice_model is None:
                return None, 0.0, "Voice model not trained"
            
            if len(self.voice_profiles) < 2:
                return None, 0.0, "Need at least 2 registered users"
            
            # Extract features
            features = self.extract_voice_features(audio_data)
            if features is None:
                return None, 0.0, "Could not extract voice features"
            
            # Handle NaN values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict
            probabilities = self.voice_model.predict_proba(features_scaled)[0]
            predicted_user_id = self.voice_model.predict(features_scaled)[0]
            confidence = max(probabilities)
            
            # Confidence threshold
            if confidence > 0.6:  # Adjustable threshold
                return predicted_user_id, confidence, "Success"
            else:
                return None, confidence, "Low confidence recognition"
                
        except Exception as e:
            return None, 0.0, f"Error in speaker identification: {e}"
    
    def speech_to_text(self, audio_obj):
        """Convert speech to text using Google Speech Recognition."""
        try:
            # Try Google Speech Recognition first
            text = self.recognizer.recognize_google(audio_obj)
            return text, "Success"
        except sr.UnknownValueError:
            return None, "Could not understand speech"
        except sr.RequestError as e:
            return None, f"Speech recognition service error: {e}"
        except Exception as e:
            return None, f"Speech to text error: {e}"
    
    def process_voice_input(self, duration=5):
        """Complete voice processing pipeline: record -> identify -> transcribe."""
        try:
            print("Listening...")
            
            # Record audio
            audio_data, audio_obj = self.record_audio_sample(duration)
            if audio_data is None:
                return None, "Failed to record audio"
            
            # Identify speaker
            user_id, confidence, id_message = self.identify_speaker_from_audio(audio_data)
            
            # Convert to text
            text, text_message = self.speech_to_text(audio_obj)
            
            result = {
                'user_id': user_id,
                'confidence': confidence,
                'text': text,
                'identification_status': id_message,
                'transcription_status': text_message,
                'audio_length': len(audio_data) / 16000  # seconds
            }
            
            return result, "Processing completed"
            
        except Exception as e:
            return None, f"Voice processing error: {e}"
    
    def get_user_info(self, user_id):
        """Get user information by ID."""
        if user_id in self.voice_profiles:
            profile = self.voice_profiles[user_id]
            return {
                'user_id': user_id,
                'name': profile['name'],
                'care_type': profile['care_type'],
                'created_at': profile['created_at'],
                'sample_count': profile['sample_count']
            }
        return None
    
    def list_registered_users(self):
        """List all registered users."""
        users = []
        for user_id, profile in self.voice_profiles.items():
            users.append({
                'user_id': user_id,
                'name': profile['name'],
                'care_type': profile['care_type'],
                'created_at': profile['created_at']
            })
        return users
    
    def delete_user_profile(self, user_id):
        """Delete a user profile and associated files."""
        try:
            if user_id not in self.voice_profiles:
                return False, "User not found"
            
            profile = self.voice_profiles[user_id]
            
            # Delete audio files
            for audio_file in profile.get('audio_files', []):
                audio_path = os.path.join(self.voice_profiles_dir, audio_file)
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            
            # Delete profile file
            profile_file = os.path.join(self.voice_profiles_dir, f"{user_id}_profile.pkl")
            if os.path.exists(profile_file):
                os.remove(profile_file)
            
            # Remove from memory
            del self.voice_profiles[user_id]
            
            # Retrain model if there are still users
            if len(self.voice_profiles) >= 2:
                self.train_voice_model()
            
            return True, "User profile deleted successfully"
            
        except Exception as e:
            return False, f"Error deleting user profile: {e}"

# Test function
def test_voice_system():
    """Test the voice recognition system."""
    voice_system = RealVoiceRecognitionSystem(
        voice_profiles_dir="/tmp/test_voice_profiles",
        model_file="/tmp/test_voice_model.pkl"
    )
    
    print("Voice Recognition System Test")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Register new user")
        print("2. Test voice recognition")
        print("3. List users")
        print("4. Delete user")
        print("5. Exit")
        
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == '1':
            name = input("Enter user name: ").strip()
            care_type = input("Enter care type (elderly/children/neurodivergent/normal/home_automation): ").strip()
            success, result = voice_system.register_voice_profile(name, care_type)
            print(f"Registration {'successful' if success else 'failed'}: {result}")
            
        elif choice == '2':
            if len(voice_system.voice_profiles) < 2:
                print("Need at least 2 registered users for testing")
                continue
                
            result, message = voice_system.process_voice_input()
            if result:
                print(f"Recognition result:")
                print(f"  User: {result.get('user_id', 'Unknown')}")
                print(f"  Confidence: {result.get('confidence', 0):.2f}")
                print(f"  Text: {result.get('text', 'No text')}")
                print(f"  Status: {result.get('identification_status', 'Unknown')}")
            else:
                print(f"Recognition failed: {message}")
                
        elif choice == '3':
            users = voice_system.list_registered_users()
            if users:
                print("Registered users:")
                for user in users:
                    print(f"  - {user['name']} ({user['care_type']}) - ID: {user['user_id'][:8]}...")
            else:
                print("No registered users")
                
        elif choice == '4':
            users = voice_system.list_registered_users()
            if not users:
                print("No users to delete")
                continue
                
            print("Select user to delete:")
            for i, user in enumerate(users, 1):
                print(f"  {i}. {user['name']}")
            
            try:
                user_idx = int(input("Enter user number: ")) - 1
                if 0 <= user_idx < len(users):
                    user_id = users[user_idx]['user_id']
                    success, message = voice_system.delete_user_profile(user_id)
                    print(message)
                else:
                    print("Invalid user number")
            except ValueError:
                print("Invalid input")
                
        elif choice == '5':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    test_voice_system()
