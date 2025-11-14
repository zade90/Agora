"""
Local runner for the fine-tuned Qwen3 comprehensive care model.
Run the model locally for inference and chat interaction with memory features.
Supports care for elderly, children, neurodivergent individuals, and home automation.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import sys
import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid

# Model path and memory storage paths
MODEL_PATH = '/hd1/Joydeep-multiagent/xagents/Fine-Models/Final-qwen3-elderly-care'
MEMORY_DIR = '/hd1/Joydeep-multiagent/memory'
USER_PROFILES_FILE = os.path.join(MEMORY_DIR, 'user_profiles.json')
CONVERSATION_HISTORY_FILE = os.path.join(MEMORY_DIR, 'conversation_history.pkl')

# Create memory directory if it doesn't exist
os.makedirs(MEMORY_DIR, exist_ok=True)

# User care types
CARE_TYPES = {
    'elderly': {
        'emoji': '🧓',
        'description': 'Elderly Care Assistant',
        'system_prompt': 'You are a compassionate AI assistant specialized in elderly care, providing practical advice for seniors and their caregivers. You help with health management, medication reminders, safety tips, social engagement, and daily living assistance.'
    },
    'children': {
        'emoji': '👶',
        'description': 'Child Care Assistant',
        'system_prompt': 'You are a friendly AI assistant specialized in child care, providing guidance for parents and caregivers. You help with child development, education, safety, nutrition, behavior management, and fun activities for kids.'
    },
    'neurodivergent': {
        'emoji': '🧠',
        'description': 'Neurodivergent Support Assistant',
        'system_prompt': 'You are an understanding AI assistant specialized in supporting neurodivergent individuals (autism, ADHD, dyslexia, etc.) and their families. You provide strategies for communication, sensory management, routine building, and creating supportive environments.'
    },
    'normal': {
        'emoji': '👤',
        'description': 'General Care Assistant',
        'system_prompt': 'You are a helpful AI assistant providing general care and home automation support. You help with daily routines, health and wellness, smart home management, productivity, and general life assistance.'
    },
    'home_automation': {
        'emoji': '🏠',
        'description': 'Smart Home Assistant',
        'system_prompt': 'You are an intelligent home automation assistant. You help manage smart devices, optimize energy usage, enhance security, automate routines, and create comfortable living environments for all household members.'
    }
}

class MemoryManager:
    """Manages conversation memory and user profiles."""
    
    def __init__(self):
        self.conversation_history = self.load_conversation_history()
        self.user_profiles = self.load_user_profiles()
    
    def load_conversation_history(self) -> Dict[str, List[Dict]]:
        """Load conversation history from file."""
        if os.path.exists(CONVERSATION_HISTORY_FILE):
            try:
                with open(CONVERSATION_HISTORY_FILE, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load conversation history: {e}")
        return {}
    
    def save_conversation_history(self):
        """Save conversation history to file."""
        try:
            with open(CONVERSATION_HISTORY_FILE, 'wb') as f:
                pickle.dump(self.conversation_history, f)
        except Exception as e:
            print(f"Warning: Could not save conversation history: {e}")
    
    def load_user_profiles(self) -> Dict[str, Dict]:
        """Load user profiles from file."""
        if os.path.exists(USER_PROFILES_FILE):
            try:
                with open(USER_PROFILES_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load user profiles: {e}")
        return {}
    
    def save_user_profiles(self):
        """Save user profiles to file."""
        try:
            with open(USER_PROFILES_FILE, 'w') as f:
                json.dump(self.user_profiles, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save user profiles: {e}")
    
    def create_user_profile(self, user_id: str, name: str, care_type: str, 
                          preferences: Dict = None, medical_info: Dict = None) -> Dict:
        """Create a new user profile."""
        profile = {
            'user_id': user_id,
            'name': name,
            'care_type': care_type,
            'created_at': datetime.now().isoformat(),
            'last_interaction': None,
            'preferences': preferences or {},
            'medical_info': medical_info or {},
            'interaction_count': 0,
            'important_notes': [],
            'routines': [],
            'reminders': []
        }
        self.user_profiles[user_id] = profile
        self.save_user_profiles()
        return profile
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile by ID."""
        return self.user_profiles.get(user_id)
    
    def update_user_profile(self, user_id: str, updates: Dict):
        """Update user profile with new information."""
        if user_id in self.user_profiles:
            self.user_profiles[user_id].update(updates)
            self.user_profiles[user_id]['last_interaction'] = datetime.now().isoformat()
            self.save_user_profiles()
    
    def add_conversation_turn(self, user_id: str, user_message: str, assistant_response: str):
        """Add a conversation turn to history."""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        turn = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'assistant_response': assistant_response
        }
        
        self.conversation_history[user_id].append(turn)
        
        # Keep only last 50 conversations per user to manage memory
        if len(self.conversation_history[user_id]) > 50:
            self.conversation_history[user_id] = self.conversation_history[user_id][-50:]
        
        self.save_conversation_history()
        
        # Update user interaction count
        if user_id in self.user_profiles:
            self.user_profiles[user_id]['interaction_count'] += 1
            self.save_user_profiles()
    
    def get_recent_conversations(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Get recent conversations for context."""
        if user_id in self.conversation_history:
            return self.conversation_history[user_id][-limit:]
        return []
    
    def add_reminder(self, user_id: str, reminder: str, reminder_time: str = None):
        """Add a reminder for the user."""
        if user_id in self.user_profiles:
            reminder_obj = {
                'id': str(uuid.uuid4()),
                'text': reminder,
                'created_at': datetime.now().isoformat(),
                'reminder_time': reminder_time,
                'completed': False
            }
            self.user_profiles[user_id]['reminders'].append(reminder_obj)
            self.save_user_profiles()

def load_model():
    """Load the fine-tuned Qwen3 comprehensive care model."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    if not torch.cuda.is_available():
        model = model.to(device)
    
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer, device

def generate_response(model, tokenizer, device, prompt, user_profile=None, memory_manager=None, 
                     max_length=512, temperature=0.7, top_p=0.9):
    """Generate a response using the fine-tuned model with memory and user context."""
    
    # Determine care type and system prompt
    if user_profile:
        care_type = user_profile.get('care_type', 'normal')
        user_name = user_profile.get('name', 'User')
        care_info = CARE_TYPES.get(care_type, CARE_TYPES['normal'])
        system_prompt = care_info['system_prompt']
        
        # Add user-specific context
        context_info = f"\nUser: {user_name} (Care type: {care_type})"
        if user_profile.get('preferences'):
            context_info += f"\nPreferences: {', '.join(user_profile['preferences'].keys())}"
        if user_profile.get('important_notes'):
            context_info += f"\nImportant notes: {'; '.join(user_profile['important_notes'][-3:])}"
        
        system_prompt += context_info
        
        # Add recent conversation context
        if memory_manager and user_profile.get('user_id'):
            recent_conversations = memory_manager.get_recent_conversations(user_profile['user_id'], 3)
            if recent_conversations:
                context_info += "\n\nRecent conversation context:"
                for conv in recent_conversations[-2:]:  # Last 2 conversations for context
                    context_info += f"\nPrevious - User: {conv['user_message'][:100]}..."
                    context_info += f"\nPrevious - Assistant: {conv['assistant_response'][:100]}..."
        
        system_prompt += context_info
    else:
        # Default to general care
        care_info = CARE_TYPES['normal']
        system_prompt = care_info['system_prompt']
    
    # Format prompt with comprehensive care context
    formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
    if torch.cuda.is_available():
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

def setup_user_profile(memory_manager):
    """Setup or select user profile for the session."""
    print("\n" + "="*60)
    print("🏠 Comprehensive Care AI Assistant - User Setup")
    print("="*60)
    
    # Show existing users
    existing_users = memory_manager.user_profiles
    if existing_users:
        print("\nExisting users:")
        for i, (user_id, profile) in enumerate(existing_users.items(), 1):
            care_info = CARE_TYPES.get(profile['care_type'], CARE_TYPES['normal'])
            print(f"  {i}. {profile['name']} ({care_info['emoji']} {care_info['description']})")
        
        print(f"  {len(existing_users) + 1}. Create new user")
        
        try:
            choice = input(f"\nSelect user (1-{len(existing_users) + 1}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(existing_users):
                user_id = list(existing_users.keys())[int(choice) - 1]
                return existing_users[user_id]
            elif choice == str(len(existing_users) + 1):
                pass  # Create new user
            else:
                print("Invalid choice. Creating new user...")
        except (ValueError, KeyboardInterrupt):
            print("Creating new user...")
    
    # Create new user
    print("\nCreating new user profile...")
    name = input("Enter user name: ").strip()
    
    print("\nSelect care type:")
    care_types = list(CARE_TYPES.keys())
    for i, care_type in enumerate(care_types, 1):
        care_info = CARE_TYPES[care_type]
        print(f"  {i}. {care_info['emoji']} {care_info['description']}")
    
    try:
        care_choice = input(f"Select care type (1-{len(care_types)}): ").strip()
        if care_choice.isdigit() and 1 <= int(care_choice) <= len(care_types):
            care_type = care_types[int(care_choice) - 1]
        else:
            care_type = 'normal'
            print("Invalid choice. Using general care.")
    except (ValueError, KeyboardInterrupt):
        care_type = 'normal'
        print("Using general care.")
    
    # Optional additional info
    preferences = {}
    medical_info = {}
    
    if care_type in ['elderly', 'neurodivergent', 'children']:
        print(f"\nOptional: Add any important notes or preferences for {name}:")
        notes = input("(Press Enter to skip): ").strip()
        if notes:
            preferences['notes'] = notes
    
    # Create profile
    user_id = str(uuid.uuid4())
    profile = memory_manager.create_user_profile(user_id, name, care_type, preferences, medical_info)
    
    print(f"\n✅ Created profile for {name}")
    return profile

def interactive_chat(model, tokenizer, device):
    """Start an interactive chat session with memory and user profiles."""
    memory_manager = MemoryManager()
    
    # Setup user profile
    user_profile = setup_user_profile(memory_manager)
    care_info = CARE_TYPES.get(user_profile['care_type'], CARE_TYPES['normal'])
    
    print("\n" + "="*60)
    print(f"{care_info['emoji']} {care_info['description']} - Interactive Chat")
    print(f"Welcome back, {user_profile['name']}!")
    print("="*60)
    print("Commands:")
    print("  'quit', 'exit', 'bye' - End conversation")
    print("  'help' - Show usage tips")
    print("  'profile' - View/edit your profile")
    print("  'memory' - View recent conversations")
    print("  'reminder <text>' - Add a reminder")
    print("  'switch' - Switch to different user")
    print("="*60 + "\n")
    
    # Show recent context if available
    recent_conversations = memory_manager.get_recent_conversations(user_profile['user_id'], 2)
    if recent_conversations:
        print("💭 Recent conversation context:")
        for conv in recent_conversations[-1:]:  # Show last conversation
            print(f"   You asked: {conv['user_message'][:80]}...")
        print()
    
    while True:
        try:
            user_input = input(f"👤 {user_profile['name']}: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"{care_info['emoji']} Goodbye, {user_profile['name']}! Take care!")
                break
            
            if user_input.lower() == 'switch':
                user_profile = setup_user_profile(memory_manager)
                care_info = CARE_TYPES.get(user_profile['care_type'], CARE_TYPES['normal'])
                print(f"\n{care_info['emoji']} Switched to {user_profile['name']}'s profile\n")
                continue
            
            if user_input.lower() == 'profile':
                print(f"\n� Profile for {user_profile['name']}:")
                print(f"   Care Type: {care_info['description']}")
                print(f"   Interactions: {user_profile.get('interaction_count', 0)}")
                print(f"   Created: {user_profile.get('created_at', 'Unknown')}")
                if user_profile.get('important_notes'):
                    print(f"   Notes: {'; '.join(user_profile['important_notes'][-3:])}")
                if user_profile.get('reminders'):
                    active_reminders = [r for r in user_profile['reminders'] if not r.get('completed')]
                    if active_reminders:
                        print(f"   Active reminders: {len(active_reminders)}")
                print()
                continue
            
            if user_input.lower() == 'memory':
                recent = memory_manager.get_recent_conversations(user_profile['user_id'], 5)
                if recent:
                    print(f"\n💭 Recent conversations for {user_profile['name']}:")
                    for i, conv in enumerate(recent[-3:], 1):
                        print(f"   {i}. You: {conv['user_message'][:60]}...")
                        print(f"      AI: {conv['assistant_response'][:60]}...")
                else:
                    print("\n💭 No previous conversations found.")
                print()
                continue
            
            if user_input.lower().startswith('reminder '):
                reminder_text = user_input[9:].strip()
                if reminder_text:
                    memory_manager.add_reminder(user_profile['user_id'], reminder_text)
                    print(f"✅ Added reminder: {reminder_text}\n")
                else:
                    print("❌ Please provide reminder text.\n")
                continue
            
            if user_input.lower() == 'help':
                print(f"\n💡 Usage Tips for {care_info['description']}:")
                if user_profile['care_type'] == 'elderly':
                    print("- Ask about health, medication, safety tips")
                    print("- Get caregiving advice and support")
                    print("- Request activities and social engagement ideas")
                elif user_profile['care_type'] == 'children':
                    print("- Ask about child development and education")
                    print("- Get parenting tips and activity suggestions")
                    print("- Request safety and nutrition advice")
                elif user_profile['care_type'] == 'neurodivergent':
                    print("- Ask about communication strategies")
                    print("- Get sensory management tips")
                    print("- Request routine and environment support")
                elif user_profile['care_type'] == 'home_automation':
                    print("- Ask about smart home setup and optimization")
                    print("- Get automation routine suggestions")
                    print("- Request security and energy management tips")
                else:
                    print("- Ask about daily routines and wellness")
                    print("- Get general life assistance and tips")
                    print("- Request home management advice")
                print()
                continue
            
            if not user_input:
                print("Please enter a question or message.")
                continue
            
            print(f"{care_info['emoji']} Assistant: ", end="", flush=True)
            response = generate_response(model, tokenizer, device, user_input, 
                                       user_profile, memory_manager)
            print(response)
            print()
            
            # Save conversation to memory
            memory_manager.add_conversation_turn(user_profile['user_id'], user_input, response)
            
        except KeyboardInterrupt:
            print(f"\n\n{care_info['emoji']} Goodbye, {user_profile['name']}! Take care!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")

def single_prompt(model, tokenizer, device, prompt, care_type='normal'):
    """Process a single prompt and return the response."""
    # Create temporary user profile for single prompt
    user_profile = {
        'name': 'User',
        'care_type': care_type,
        'user_id': 'temp_user',
        'preferences': {},
        'important_notes': []
    }
    
    response = generate_response(model, tokenizer, device, prompt, user_profile)
    care_info = CARE_TYPES.get(care_type, CARE_TYPES['normal'])
    print(f"\n{care_info['emoji']} {care_info['description']}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    return response

def main():
    parser = argparse.ArgumentParser(description="Run the fine-tuned Qwen3 comprehensive care model locally")
    parser.add_argument("--prompt", "-p", type=str, help="Single prompt to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--care_type", "-c", type=str, choices=list(CARE_TYPES.keys()), 
                       default='normal', help="Care type for single prompt mode")
    parser.add_argument("--max_length", "-m", type=int, default=512, help="Maximum response length")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--list_users", action="store_true", help="List existing user profiles")
    
    args = parser.parse_args()
    
    # List users if requested
    if args.list_users:
        memory_manager = MemoryManager()
        if memory_manager.user_profiles:
            print("\n👥 Existing User Profiles:")
            print("="*40)
            for user_id, profile in memory_manager.user_profiles.items():
                care_info = CARE_TYPES.get(profile['care_type'], CARE_TYPES['normal'])
                last_interaction = profile.get('last_interaction', 'Never')
                if last_interaction != 'Never':
                    last_interaction = datetime.fromisoformat(last_interaction).strftime('%Y-%m-%d %H:%M')
                print(f"{care_info['emoji']} {profile['name']} ({care_info['description']})")
                print(f"   Last interaction: {last_interaction}")
                print(f"   Total interactions: {profile.get('interaction_count', 0)}")
                if profile.get('reminders'):
                    active_reminders = [r for r in profile['reminders'] if not r.get('completed')]
                    if active_reminders:
                        print(f"   Active reminders: {len(active_reminders)}")
                print()
        else:
            print("\n👥 No user profiles found.")
        return
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
    
    # Load model
    try:
        model, tokenizer, device = load_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Process based on arguments
    if args.prompt:
        # Single prompt mode
        single_prompt(model, tokenizer, device, args.prompt, args.care_type)
    elif args.interactive:
        # Interactive chat mode
        interactive_chat(model, tokenizer, device)
    else:
        # Default: show example and start interactive mode
        print("\n🏠 Comprehensive Care AI Assistant")
        print("="*50)
        print("Supporting: Elderly, Children, Neurodivergent, General Care & Home Automation")
        print("\nExample usage:")
        print("  python local-run.py -p 'Help me set up medication reminders' -c elderly")
        print("  python local-run.py -p 'Fun activities for my 8-year-old' -c children")
        print("  python local-run.py -p 'Managing sensory overload' -c neurodivergent")
        print("  python local-run.py -p 'Automate my morning routine' -c home_automation")
        print("  python local-run.py -i  # Interactive mode with user profiles")
        print("  python local-run.py --list_users  # Show existing users")
        print("\nStarting interactive mode...\n")
        interactive_chat(model, tokenizer, device)

if __name__ == "__main__":
    main()
