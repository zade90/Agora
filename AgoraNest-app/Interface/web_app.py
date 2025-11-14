"""
Web-based interface for the Comprehensive Care AI Assistant with voice recognition and memory.
Features:
- Voice-based user registration and authentication
- Speech-to-text and text-to-speech functionality
- Memory management with CSV logging
- Actionable responses and task execution
"""
import os
import csv
import json
import pickle
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import time

# Web framework
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit

# Voice and audio processing
import speech_recognition as sr
import pyttsx3
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# AI model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_PATH = '/hd1/Joydeep-multiagent/xagents/Fine-Models/Final-qwen3-elderly-care'
MEMORY_DIR = '/hd1/Joydeep-multiagent/memory'
WEB_DATA_DIR = '/hd1/Joydeep-multiagent/Interface/web_data'
VOICE_PROFILES_DIR = os.path.join(WEB_DATA_DIR, 'voice_profiles')
CONVERSATION_LOG_FILE = os.path.join(WEB_DATA_DIR, 'conversation_log.csv')
VOICE_MODEL_FILE = os.path.join(WEB_DATA_DIR, 'voice_recognition_model.pkl')
USER_PROFILES_FILE = os.path.join(MEMORY_DIR, 'user_profiles.json')

# Create directories
os.makedirs(WEB_DATA_DIR, exist_ok=True)
os.makedirs(VOICE_PROFILES_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)

# Care types
CARE_TYPES = {
    'elderly': {
        'emoji': '🧓',
        'description': 'Elderly Care Assistant',
        'system_prompt': 'You are a compassionate AI assistant specialized in elderly care, providing practical advice for seniors and their caregivers. You help with health management, medication reminders, safety tips, social engagement, and daily living assistance.',
        'actionable_keywords': ['reminder', 'medicine', 'appointment', 'call', 'schedule', 'alert', 'emergency']
    },
    'children': {
        'emoji': '👶',
        'description': 'Child Care Assistant',
        'system_prompt': 'You are a friendly AI assistant specialized in child care, providing guidance for parents and caregivers. You help with child development, education, safety, nutrition, behavior management, and fun activities for kids.',
        'actionable_keywords': ['schedule', 'play', 'learn', 'bedtime', 'meal', 'activity', 'game']
    },
    'neurodivergent': {
        'emoji': '🧠',
        'description': 'Neurodivergent Support Assistant',
        'system_prompt': 'You are an understanding AI assistant specialized in supporting neurodivergent individuals and their families. You provide strategies for communication, sensory management, routine building, and creating supportive environments.',
        'actionable_keywords': ['routine', 'schedule', 'reminder', 'calm', 'sensory', 'break', 'timer']
    },
    'normal': {
        'emoji': '👤',
        'description': 'General Care Assistant',
        'system_prompt': 'You are a helpful AI assistant providing general care and home automation support. You help with daily routines, health and wellness, smart home management, productivity, and general life assistance.',
        'actionable_keywords': ['reminder', 'schedule', 'task', 'home', 'automation', 'control']
    },
    'home_automation': {
        'emoji': '🏠',
        'description': 'Smart Home Assistant',
        'system_prompt': 'You are an intelligent home automation assistant. You help manage smart devices, optimize energy usage, enhance security, automate routines, and create comfortable living environments.',
        'actionable_keywords': ['lights', 'temperature', 'security', 'automation', 'control', 'device', 'schedule']
    }
}

class VoiceRecognitionSystem:
    """Handles voice recognition and user identification."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.voice_profiles = {}
        self.audio_available = False
        
        # Try to initialize microphone, but continue without it if not available
        try:
            self.microphone = sr.Microphone()
            # Test microphone access
            with self.microphone as source:
                print("🎤 Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("✅ Voice recognition system ready")
            self.audio_available = True
        except (OSError, Exception) as e:
            print(f"⚠️  Audio device not available: {e}")
            print("🔧 Running in text-only mode. Voice features will be disabled.")
            self.audio_available = False
        
        self.load_voice_profiles()
    
    def extract_voice_features(self, audio_data):
        """Extract voice features for recognition using librosa."""
        try:
            # Convert audio data to numpy array
            audio_np = np.frombuffer(audio_data.get_wav_data(), dtype=np.int16).astype(np.float32)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio_np, sr=16000, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Extract additional features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_np, sr=16000)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_np, sr=16000)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_np)
            
            # Combine features
            features = np.concatenate([
                mfcc_mean, mfcc_std,
                np.mean(spectral_centroids), np.std(spectral_centroids),
                np.mean(spectral_rolloff), np.std(spectral_rolloff),
                np.mean(zero_crossing_rate), np.std(zero_crossing_rate)
            ])
            
            return features
        except Exception as e:
            print(f"Error extracting voice features: {e}")
            return None
    
    def register_voice(self, user_name, audio_samples):
        """Register a new voice profile with multiple audio samples."""
        try:
            features_list = []
            
            for audio_data in audio_samples:
                features = self.extract_voice_features(audio_data)
                if features is not None:
                    features_list.append(features)
            
            if len(features_list) < 3:
                return False, "Need at least 3 voice samples for registration"
            
            # Store voice profile
            user_id = str(uuid.uuid4())
            self.voice_profiles[user_id] = {
                'name': user_name,
                'features': features_list,
                'created_at': datetime.now().isoformat()
            }
            
            # Save voice profile
            profile_file = os.path.join(VOICE_PROFILES_DIR, f"{user_id}.pkl")
            with open(profile_file, 'wb') as f:
                pickle.dump(self.voice_profiles[user_id], f)
            
            # Retrain voice recognition model
            self.train_voice_model()
            
            return True, user_id
            
        except Exception as e:
            return False, f"Error registering voice: {e}"
    
    def train_voice_model(self):
        """Train/retrain the voice recognition model with all profiles."""
        try:
            if len(self.voice_profiles) < 2:
                return
            
            X = []
            y = []
            
            for user_id, profile in self.voice_profiles.items():
                for features in profile['features']:
                    X.append(features)
                    y.append(user_id)
            
            X = np.array(X)
            X_scaled = self.scaler.fit_transform(X)
            
            # Train SVM classifier
            self.voice_model = SVC(kernel='rbf', probability=True)
            self.voice_model.fit(X_scaled, y)
            
            # Save model
            model_data = {
                'model': self.voice_model,
                'scaler': self.scaler,
                'user_mapping': {uid: profile['name'] for uid, profile in self.voice_profiles.items()}
            }
            
            with open(VOICE_MODEL_FILE, 'wb') as f:
                pickle.dump(model_data, f)
                
        except Exception as e:
            print(f"Error training voice model: {e}")
    
    def load_voice_model(self):
        """Load the trained voice recognition model."""
        try:
            if os.path.exists(VOICE_MODEL_FILE):
                with open(VOICE_MODEL_FILE, 'rb') as f:
                    model_data = pickle.load(f)
                    self.voice_model = model_data['model']
                    self.scaler = model_data['scaler']
        except Exception as e:
            print(f"Error loading voice model: {e}")
    
    def load_voice_profiles(self):
        """Load existing voice profiles."""
        try:
            for filename in os.listdir(VOICE_PROFILES_DIR):
                if filename.endswith('.pkl'):
                    user_id = filename[:-4]
                    with open(os.path.join(VOICE_PROFILES_DIR, filename), 'rb') as f:
                        self.voice_profiles[user_id] = pickle.load(f)
        except Exception as e:
            print(f"Error loading voice profiles: {e}")
    
    def identify_speaker(self, audio_data):
        """Identify speaker from audio data."""
        try:
            if self.voice_model is None or len(self.voice_profiles) < 2:
                return None, 0.0
            
            features = self.extract_voice_features(audio_data)
            if features is None:
                return None, 0.0
            
            features_scaled = self.scaler.transform([features])
            probabilities = self.voice_model.predict_proba(features_scaled)[0]
            predicted_user_id = self.voice_model.predict(features_scaled)[0]
            confidence = max(probabilities)
            
            if confidence > 0.7:  # Confidence threshold
                return predicted_user_id, confidence
            else:
                return None, confidence
                
        except Exception as e:
            print(f"Error identifying speaker: {e}")
            return None, 0.0
    
    def speech_to_text(self, audio_data):
        """Convert speech to text using speech recognition."""
        if not self.audio_available:
            return None
            
        try:
            text = self.recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None

class TextToSpeechEngine:
    """Handles text-to-speech conversion."""
    
    def __init__(self):
        self.engine = None
        self.audio_available = False
        
        try:
            self.engine = pyttsx3.init()
            self.setup_voice()
            self.audio_available = True
            print("✅ Text-to-speech engine ready")
        except Exception as e:
            print(f"⚠️  TTS engine not available: {e}")
            print("🔧 Text-to-speech will be disabled.")
            self.audio_available = False
    
    def setup_voice(self):
        """Setup TTS voice properties."""
        if not self.engine:
            return
            
        try:
            voices = self.engine.getProperty('voices')
            if voices:
                # Prefer female voice for care assistant
                for voice in voices:
                    if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            self.engine.setProperty('rate', 150)  # Speed
            self.engine.setProperty('volume', 0.9)  # Volume
        except Exception as e:
            print(f"⚠️  Voice setup failed: {e}")
    
    def speak(self, text):
        """Convert text to speech."""
        if not self.audio_available or not self.engine:
            return False
            
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"TTS error: {e}")
            return False
        except Exception as e:
            print(f"TTS error: {e}")

class ConversationLogger:
    """Handles conversation logging to CSV."""
    
    def __init__(self):
        self.ensure_csv_exists()
    
    def ensure_csv_exists(self):
        """Ensure CSV file exists with proper headers."""
        if not os.path.exists(CONVERSATION_LOG_FILE):
            with open(CONVERSATION_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'user_id', 'user_name', 'care_type', 
                    'input_text', 'output_text', 'confidence', 'action_taken'
                ])
    
    def log_conversation(self, user_id, user_name, care_type, input_text, 
                        output_text, confidence=0.0, action_taken=None):
        """Log conversation to CSV."""
        try:
            with open(CONVERSATION_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    user_id,
                    user_name,
                    care_type,
                    input_text,
                    output_text,
                    confidence,
                    action_taken or ''
                ])
        except Exception as e:
            print(f"Error logging conversation: {e}")

class ActionableResponseSystem:
    """Handles actionable responses and task execution."""
    
    def __init__(self):
        self.actions = {
            'reminder': self.set_reminder,
            'schedule': self.schedule_task,
            'call': self.make_call,
            'emergency': self.emergency_alert,
            'control': self.home_control,
            'automation': self.home_automation
        }
    
    def detect_actionable_intent(self, text, care_type):
        """Detect if the text contains actionable intent."""
        text_lower = text.lower()
        keywords = CARE_TYPES.get(care_type, {}).get('actionable_keywords', [])
        
        for keyword in keywords:
            if keyword in text_lower:
                return keyword
        return None
    
    def execute_action(self, action_type, text, user_id, user_name):
        """Execute the identified action."""
        if action_type in self.actions:
            return self.actions[action_type](text, user_id, user_name)
        return None
    
    def set_reminder(self, text, user_id, user_name):
        """Set a reminder for the user."""
        # Extract time and reminder text (simplified)
        reminder_text = text
        return f"✅ Reminder set for {user_name}: {reminder_text}"
    
    def schedule_task(self, text, user_id, user_name):
        """Schedule a task."""
        return f"📅 Task scheduled for {user_name}: {text}"
    
    def make_call(self, text, user_id, user_name):
        """Simulate making a call (placeholder)."""
        return f"📞 Call request noted for {user_name}"
    
    def emergency_alert(self, text, user_id, user_name):
        """Handle emergency alert."""
        return f"🚨 Emergency alert activated for {user_name}"
    
    def home_control(self, text, user_id, user_name):
        """Control home devices (placeholder)."""
        return f"🏠 Home control command executed for {user_name}"
    
    def home_automation(self, text, user_id, user_name):
        """Handle home automation."""
        return f"🤖 Automation routine set for {user_name}"

class ComprehensiveCareAI:
    """Main AI system integrating all components."""
    
    def __init__(self):
        self.voice_system = VoiceRecognitionSystem()
        self.tts_engine = TextToSpeechEngine()
        self.conversation_logger = ConversationLogger()
        self.action_system = ActionableResponseSystem()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.user_profiles = self.load_user_profiles()
        self.load_ai_model()
    
    def load_user_profiles(self):
        """Load user profiles from JSON file."""
        try:
            if os.path.exists(USER_PROFILES_FILE):
                with open(USER_PROFILES_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading user profiles: {e}")
        return {}
    
    def save_user_profiles(self):
        """Save user profiles to JSON file."""
        try:
            with open(USER_PROFILES_FILE, 'w') as f:
                json.dump(self.user_profiles, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving user profiles: {e}")
    
    def load_ai_model(self):
        """Load the AI model."""
        try:
            print("Loading AI model...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print("AI model loaded successfully!")
        except Exception as e:
            print(f"Error loading AI model: {e}")
    
    def generate_response(self, prompt, user_profile, max_length=512):
        """Generate AI response."""
        try:
            care_type = user_profile.get('care_type', 'normal')
            care_info = CARE_TYPES.get(care_type, CARE_TYPES['normal'])
            system_prompt = care_info['system_prompt']
            
            # Add user context
            user_name = user_profile.get('name', 'User')
            context_info = f"\nUser: {user_name} (Care type: {care_type})"
            system_prompt += context_info
            
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {e}"
    
    def process_voice_input(self, audio_data):
        """Process voice input and return response."""
        if not self.voice_system.audio_available:
            return None, "Voice input not available. Audio device not found."
            
        try:
            # Convert speech to text
            text = self.voice_system.speech_to_text(audio_data)
            if not text:
                return None, "Sorry, I couldn't understand what you said."
            
            # Identify speaker
            user_id, confidence = self.voice_system.identify_speaker(audio_data)
            
            if user_id and user_id in self.voice_system.voice_profiles:
                user_name = self.voice_system.voice_profiles[user_id]['name']
                user_profile = self.user_profiles.get(user_id, {
                    'name': user_name,
                    'care_type': 'normal',
                    'user_id': user_id
                })
            else:
                # Unknown speaker
                return None, "I don't recognize your voice. Please register first."
            
            # Generate AI response
            ai_response = self.generate_response(text, user_profile)
            
            # Check for actionable intent
            care_type = user_profile.get('care_type', 'normal')
            action_type = self.action_system.detect_actionable_intent(text, care_type)
            action_result = None
            
            if action_type:
                action_result = self.action_system.execute_action(action_type, text, user_id, user_name)
                if action_result:
                    ai_response += f"\n\n{action_result}"
            
            # Log conversation
            self.conversation_logger.log_conversation(
                user_id, user_name, care_type, text, ai_response, confidence, action_result
            )
            
            return {
                'user_id': user_id,
                'user_name': user_name,
                'input_text': text,
                'output_text': ai_response,
                'confidence': confidence,
                'action_taken': action_result
            }, ai_response
            
        except Exception as e:
            return None, f"Error processing voice input: {e}"
    
    def register_user_text(self, user_id, user_name, care_type):
        """Register user without voice data (text-only mode)."""
        try:
            # Create user profile
            profile = {
                'user_id': user_id,
                'name': user_name,
                'care_type': care_type,
                'created_at': datetime.now().isoformat(),
                'preferences': {},
                'medical_info': {},
                'interaction_count': 0,
                'last_interaction': None,
                'voice_enabled': False
            }
            
            self.user_profiles[user_id] = profile
            self.save_user_profiles()
            
            print(f"✅ User {user_name} registered in text-only mode")
            return True
            
        except Exception as e:
            print(f"❌ Text registration failed: {e}")
            return False
    
    def load_user_profiles(self):
        """Load user profiles from file."""
        try:
            if os.path.exists(os.path.join(MEMORY_DIR, 'user_profiles.json')):
                with open(os.path.join(MEMORY_DIR, 'user_profiles.json'), 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading user profiles: {e}")
        return {}
    
    def save_user_profiles(self):
        """Save user profiles to file."""
        try:
            os.makedirs(MEMORY_DIR, exist_ok=True)
            with open(os.path.join(MEMORY_DIR, 'user_profiles.json'), 'w') as f:
                json.dump(self.user_profiles, f, indent=2)
        except Exception as e:
            print(f"Error saving user profiles: {e}")

# Flask application (serve the local `assets` folder at /assets)
# static_folder is relative to this module's directory
app = Flask(__name__, static_folder='assets', static_url_path='/assets')
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize AI system
ai_system = ComprehensiveCareAI()

@app.route('/')
def index():
    """Main page."""
    audio_status = {
        'voice_available': ai_system.voice_system.audio_available,
        'tts_available': ai_system.tts_engine.audio_available
    }
    return render_template('index.html', audio_status=audio_status)

@app.route('/register')
def register():
    """User registration page."""
    audio_status = {
        'voice_available': ai_system.voice_system.audio_available,
        'tts_available': ai_system.tts_engine.audio_available
    }
    return render_template('register.html', audio_status=audio_status)

@app.route('/api/audio_status')
def audio_status():
    """Get audio system status."""
    return jsonify({
        'voice_available': ai_system.voice_system.audio_available,
        'tts_available': ai_system.tts_engine.audio_available,
        'message': 'Audio system ready' if ai_system.voice_system.audio_available else 'Running in text-only mode'
    })

@app.route('/api/register_text', methods=['POST'])
def register_text():
    """Register user with text-based input (fallback when no audio)."""
    try:
        data = request.get_json()
        user_name = data.get('name', '').strip()
        care_type = data.get('care_type', 'normal')
        
        if not user_name:
            return jsonify({'success': False, 'message': 'Name is required'})
        
        # Create user profile without voice data
        user_id = str(uuid.uuid4())
        success = ai_system.register_user_text(user_id, user_name, care_type)
        
        if success:
            return jsonify({
                'success': True, 
                'message': f'User {user_name} registered successfully (text-only mode)',
                'user_id': user_id
            })
        else:
            return jsonify({'success': False, 'message': 'Registration failed'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@socketio.on('register_voice')
def handle_voice_registration(data):
    """Handle voice registration."""
    try:
        user_name = data['name']
        care_type = data['care_type']
        
        # This would handle multiple audio samples in a real implementation
        # For now, we'll simulate successful registration
        user_id = str(uuid.uuid4())
        
        # Create user profile
        user_profile = {
            'user_id': user_id,
            'name': user_name,
            'care_type': care_type,
            'created_at': datetime.now().isoformat(),
            'voice_registered': True
        }
        
        ai_system.user_profiles[user_id] = user_profile
        ai_system.save_user_profiles()
        
        emit('registration_result', {
            'success': True,
            'message': f'Successfully registered {user_name} with {CARE_TYPES[care_type]["description"]}',
            'user_id': user_id
        })
        
    except Exception as e:
        emit('registration_result', {
            'success': False,
            'message': f'Registration failed: {e}'
        })

@socketio.on('voice_input')
def handle_voice_input(data):
    """Handle voice input for conversation."""
    try:
        # This would process actual audio data in a real implementation
        # For now, we'll simulate with text input
        text_input = data.get('text', '')
        user_id = data.get('user_id')
        
        if user_id and user_id in ai_system.user_profiles:
            user_profile = ai_system.user_profiles[user_id]
            response = ai_system.generate_response(text_input, user_profile)
            
            # Log conversation
            ai_system.conversation_logger.log_conversation(
                user_id, user_profile['name'], user_profile['care_type'],
                text_input, response
            )
            
            emit('voice_response', {
                'user_name': user_profile['name'],
                'input_text': text_input,
                'output_text': response,
                'care_type': user_profile['care_type']
            })
        else:
            emit('voice_response', {
                'error': 'User not found or not registered'
            })
            
    except Exception as e:
        emit('voice_response', {
            'error': f'Error processing input: {e}'
        })

@app.route('/api/users')
def get_users():
    """Get list of registered users."""
    users = []
    for user_id, profile in ai_system.user_profiles.items():
        care_info = CARE_TYPES.get(profile['care_type'], CARE_TYPES['normal'])
        users.append({
            'user_id': user_id,
            'name': profile['name'],
            'care_type': profile['care_type'],
            'care_description': care_info['description'],
            'emoji': care_info['emoji']
        })
    return jsonify(users)

@app.route('/api/conversation_log')
def get_conversation_log():
    """Get conversation log."""
    try:
        conversations = []
        if os.path.exists(CONVERSATION_LOG_FILE):
            with open(CONVERSATION_LOG_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                conversations = list(reader)
        return jsonify(conversations)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    import socket
    def get_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't have to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    server_ip = get_ip()
    print("🏠 Starting Comprehensive Care AI Web Interface...")
    print(f"Voice profiles directory: {VOICE_PROFILES_DIR}")
    print(f"Conversation log: {CONVERSATION_LOG_FILE}")
    print(f"Navigate to http://{server_ip}:5001 to access the interface from your browser.")
    print("If running on a remote server, use the server's IP address, not localhost.")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
