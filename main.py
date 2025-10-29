"""
Flappy Bird Co-Creation Sandbox - Main Controller
Manages the interaction loop between user, AI, and game engine.
"""

import json
import os
from game_engine import FlappyGame
from ai_co_creator import generate_level

def load_config(filepath="config.json"):
    """Load game configuration from JSON file."""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read().strip()
                if content:  # Check if file is not empty
                    return json.loads(content)
        # If file doesn't exist or is empty, return default
        return get_default_config()
    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️  Config file corrupted, using defaults. Error: {e}")
        return get_default_config()

def save_config(config, filepath="config.json"):
    """Save current configuration to JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"⚠️  Could not save config: {e}")

def get_default_config():
    """Return default game configuration."""
    return {
        "gravity": 0.5,
        "jump_strength": -10,
        "pipe_speed": 3,
        "pipe_gap": 200,
        "pipe_frequency": 90,
        "background_color": [135, 206, 235],
        "bird_color": [255, 255, 0],
        "pipe_color": [34, 139, 34],
        "difficulty": "medium"
    }

def print_banner():
    """Display welcome banner."""
    print("\n" + "="*60)
    print("🐦 FLAPPY BIRD CO-CREATION SANDBOX 🤖")
    print("="*60)
    print("Collaborate with AI to design your perfect Flappy Bird!")
    print("\nCommands:")
    print("  - Type a prompt (e.g., 'make it harder', 'change background to red')")
    print("  - Type 'play' to start the game with current settings")
    print("  - Type 'reset' to restore default configuration")
    print("  - Type 'show' to see current configuration")
    print("  - Type 'quit' to exit")
    print("="*60 + "\n")

def show_config(config):
    """Display current configuration in readable format."""
    print("\n📋 Current Configuration:")
    print(f"  Difficulty: {config['difficulty']}")
    print(f"  Gravity: {config['gravity']}")
    print(f"  Jump Strength: {config['jump_strength']}")
    print(f"  Pipe Speed: {config['pipe_speed']}")
    print(f"  Pipe Gap: {config['pipe_gap']} pixels")
    print(f"  Pipe Frequency: {config['pipe_frequency']} frames")
    print(f"  Background Color: RGB{tuple(config['background_color'])}")
    print(f"  Bird Color: RGB{tuple(config['bird_color'])}")
    print(f"  Pipe Color: RGB{tuple(config['pipe_color'])}\n")

def print_tips():
    """Display usage tips."""
    print("\n💡 Pro Tips:")
    print("  • Be specific: 'make pipes faster' works better than 'change speed'")
    print("  • Try themes: 'night mode', 'sunset', 'ocean theme', 'space'")
    print("  • Combine changes: 'make it harder and change background to dark'")
    print("  • Test often: Type 'play' after each change to feel the difference")
    print()

def main():
    """Main interaction loop."""
    print_banner()
    
    # Load or create initial configuration
    config = load_config()
    save_config(config)
    
    print("✅ Loaded initial configuration. Type 'show' to view or 'play' to start!")
    print_tips()
    
    while True:
        try:
            user_input = input("💬 Your prompt: ").strip()
        except EOFError:
            print("\n\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
            
        user_lower = user_input.lower()
            
        if user_lower == "quit" or user_lower == "exit":
            print("\n👋 Thanks for co-creating! Goodbye!")
            break
            
        elif user_lower == "play" or user_lower == "start":
            print("\n🎮 Starting game with current configuration...")
            print("   Press SPACE to jump, ESC to return to menu\n")
            try:
                game = FlappyGame(config)
                score = game.run()
                print(f"\n🏆 Game Over! Final Score: {score}")
            except Exception as e:
                print(f"\n❌ Error running game: {e}")
                import traceback
                traceback.print_exc()
            
        elif user_lower == "reset" or user_lower == "default":
            config = get_default_config()
            save_config(config)
            print("\n🔄 Configuration reset to defaults!")
            show_config(config)
            
        elif user_lower == "show" or user_lower == "config" or user_lower == "status":
            show_config(config)
            
        elif user_lower == "help" or user_lower == "tips":
            print_tips()
            
        else:
            print("\n🤖 AI is processing your request...")
            new_config = generate_level(user_input, config)
            
            if new_config:
                config = new_config
                save_config(config)
                print("✨ Configuration updated!")
                show_config(config)
            else:
                print("❌ Could not process that request. Try being more specific!")
                print("   Examples: 'make it harder', 'change background to blue', 'faster pipes'")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()