"""
AI Co-Creator Module
Converts natural language prompts into game configuration changes.
Uses Ollama for local, free, open-source AI processing.
"""

import os
import json
import re

# Check if Ollama is available
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Warning: ollama package not found. Install with: pip install ollama")
    print("   Falling back to rule-based system.")

def parse_color(color_text):
    """Convert color names to RGB values."""
    colors = {
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
        'yellow': [255, 255, 0],
        'orange': [255, 165, 0],
        'purple': [128, 0, 128],
        'pink': [255, 192, 203],
        'cyan': [0, 255, 255],
        'white': [255, 255, 255],
        'black': [0, 0, 0],
        'gray': [128, 128, 128],
        'grey': [128, 128, 128],
        'brown': [139, 69, 19],
        'skyblue': [135, 206, 235],
        'darkgreen': [34, 139, 34],
        'lightblue': [173, 216, 230],
        'darkblue': [0, 0, 139],
        'lime': [0, 255, 0],
        'magenta': [255, 0, 255],
        'maroon': [128, 0, 0],
        'navy': [0, 0, 128],
        'olive': [128, 128, 0],
        'teal': [0, 128, 128],
        'silver': [192, 192, 192],
        'gold': [255, 215, 0],
        'coral': [255, 127, 80],
        'turquoise': [64, 224, 208]
    }
    return colors.get(color_text.lower(), None)

def rule_based_generator(prompt, current_config):
    """
    Fallback rule-based system when Ollama is unavailable.
    Parses simple commands and modifies configuration.
    """
    config = current_config.copy()
    prompt_lower = prompt.lower()
    
    # Difficulty adjustments
    if any(word in prompt_lower for word in ['harder', 'difficult', 'increase difficulty', 
                                               'more challenging', 'make hard']):
        config['gravity'] = min(config['gravity'] + 0.15, 1.5)
        config['pipe_speed'] = min(config['pipe_speed'] + 1, 8)
        config['pipe_gap'] = max(config['pipe_gap'] - 30, 120)
        config['pipe_frequency'] = max(config['pipe_frequency'] - 15, 60)
        config['difficulty'] = 'hard'
        return config
        
    if any(word in prompt_lower for word in ['easier', 'easy', 'decrease difficulty', 
                                               'less challenging', 'simple', 'make easy']):
        config['gravity'] = max(config['gravity'] - 0.15, 0.2)
        config['pipe_speed'] = max(config['pipe_speed'] - 1, 1)
        config['pipe_gap'] = min(config['pipe_gap'] + 30, 300)
        config['pipe_frequency'] = min(config['pipe_frequency'] + 15, 150)
        config['difficulty'] = 'easy'
        return config
    
    # Medium difficulty
    if 'medium' in prompt_lower or 'normal' in prompt_lower:
        config['gravity'] = 0.5
        config['pipe_speed'] = 3
        config['pipe_gap'] = 200
        config['pipe_frequency'] = 90
        config['difficulty'] = 'medium'
        return config
        
    # Speed adjustments
    if any(word in prompt_lower for word in ['faster', 'speed up', 'quick', 'rapid', 
                                               'increase speed']):
        if 'pipe' in prompt_lower or 'pipes' in prompt_lower:
            config['pipe_speed'] = min(config['pipe_speed'] + 1.5, 10)
        else:
            config['pipe_speed'] = min(config['pipe_speed'] + 1.5, 10)
        return config
        
    if any(word in prompt_lower for word in ['slower', 'slow down', 'reduce speed', 
                                               'decrease speed']):
        if 'pipe' in prompt_lower or 'pipes' in prompt_lower:
            config['pipe_speed'] = max(config['pipe_speed'] - 1.5, 1)
        else:
            config['pipe_speed'] = max(config['pipe_speed'] - 1.5, 1)
        return config
        
    # Gravity adjustments
    if any(word in prompt_lower for word in ['more gravity', 'heavier', 'increase gravity', 
                                               'stronger gravity']):
        config['gravity'] = min(config['gravity'] + 0.2, 2.0)
        return config
        
    if any(word in prompt_lower for word in ['less gravity', 'lighter', 'decrease gravity', 
                                               'floaty', 'weaker gravity']):
        config['gravity'] = max(config['gravity'] - 0.2, 0.1)
        return config
        
    # Gap adjustments
    if any(word in prompt_lower for word in ['bigger gap', 'wider gap', 'larger gap', 
                                               'more space', 'increase gap']):
        config['pipe_gap'] = min(config['pipe_gap'] + 40, 350)
        return config
        
    if any(word in prompt_lower for word in ['smaller gap', 'narrower gap', 'tighter gap', 
                                               'less space', 'decrease gap']):
        config['pipe_gap'] = max(config['pipe_gap'] - 40, 100)
        return config
        
    # Jump strength adjustments
    if any(word in prompt_lower for word in ['stronger jump', 'higher jump', 'more jump', 
                                               'increase jump']):
        config['jump_strength'] = max(config['jump_strength'] - 2, -15)
        return config
        
    if any(word in prompt_lower for word in ['weaker jump', 'lower jump', 'less jump', 
                                               'decrease jump']):
        config['jump_strength'] = min(config['jump_strength'] + 2, -5)
        return config
    
    # Pipe frequency adjustments
    if any(word in prompt_lower for word in ['more pipes', 'frequent pipes', 'more obstacles']):
        config['pipe_frequency'] = max(config['pipe_frequency'] - 20, 60)
        return config
        
    if any(word in prompt_lower for word in ['fewer pipes', 'less pipes', 'less frequent']):
        config['pipe_frequency'] = min(config['pipe_frequency'] + 20, 150)
        return config
        
    # Color changes - check for specific targets
    color_found = False
    for color_key in ['background', 'bird', 'pipe']:
        if color_key in prompt_lower:
            # Extract color name
            for word in prompt_lower.split():
                rgb = parse_color(word)
                if rgb:
                    config[f'{color_key}_color'] = rgb
                    color_found = True
                    return config
    
    # If no specific target, look for color mentions and apply to background
    if not color_found:
        for word in prompt_lower.split():
            rgb = parse_color(word)
            if rgb and 'background' in prompt_lower:
                config['background_color'] = rgb
                return config
                    
    # Theme presets
    if any(word in prompt_lower for word in ['night', 'dark', 'evening', 'midnight']):
        config['background_color'] = [25, 25, 50]
        config['pipe_color'] = [50, 50, 70]
        config['bird_color'] = [255, 215, 0]
        return config
        
    if any(word in prompt_lower for word in ['day', 'bright', 'morning', 'sunny']):
        config['background_color'] = [135, 206, 235]
        config['pipe_color'] = [34, 139, 34]
        config['bird_color'] = [255, 255, 0]
        return config
        
    if 'sunset' in prompt_lower or 'dusk' in prompt_lower:
        config['background_color'] = [255, 140, 100]
        config['pipe_color'] = [100, 60, 100]
        config['bird_color'] = [255, 200, 50]
        return config
        
    if any(word in prompt_lower for word in ['ocean', 'underwater', 'sea', 'aquatic']):
        config['background_color'] = [0, 105, 148]
        config['pipe_color'] = [46, 139, 87]
        config['bird_color'] = [255, 140, 0]
        return config
        
    if any(word in prompt_lower for word in ['space', 'cosmic', 'galaxy', 'universe']):
        config['background_color'] = [10, 10, 30]
        config['pipe_color'] = [100, 100, 150]
        config['bird_color'] = [255, 255, 255]
        return config
    
    if any(word in prompt_lower for word in ['forest', 'jungle', 'nature', 'green']):
        config['background_color'] = [144, 238, 144]
        config['pipe_color'] = [34, 139, 34]
        config['bird_color'] = [255, 0, 0]
        return config
    
    if any(word in prompt_lower for word in ['desert', 'sand', 'hot']):
        config['background_color'] = [237, 201, 175]
        config['pipe_color'] = [160, 82, 45]
        config['bird_color'] = [139, 69, 19]
        return config
        
    return None

def generate_level_with_ollama(prompt, current_config):
    """
    Use Ollama (local LLM) to generate configuration.
    Requires Ollama to be installed and running locally.
    Free and open-source: https://ollama.ai
    """
    try:
        # System prompt that teaches the AI how to modify configs
        system_prompt = """You are a game design AI assistant that helps modify Flappy Bird game parameters.

Current configuration format:
{
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

Parameter constraints:
- gravity: 0.1 to 2.0 (higher = bird falls faster)
- jump_strength: -5 to -15 (more negative = jumps higher)
- pipe_speed: 1 to 10 (pixels per frame)
- pipe_gap: 100 to 350 (pixels between top and bottom pipes)
- pipe_frequency: 60 to 150 (frames between new pipes, lower = more frequent)
- background_color: RGB array [R, G, B] where each value is 0-255
- bird_color: RGB array [R, G, B]
- pipe_color: RGB array [R, G, B]
- difficulty: string ("easy", "medium", or "hard")

Your task: Given a user request, modify the appropriate parameters and return ONLY a valid JSON object with ALL fields.

Examples:
- "make it harder" → increase gravity, increase pipe_speed, decrease pipe_gap, decrease pipe_frequency, set difficulty to "hard"
- "change background to red" → set background_color to [255, 0, 0]
- "slower pipes" → decrease pipe_speed
- "bigger gap" → increase pipe_gap

Return ONLY the JSON object, nothing else."""

        # Prepare user message with current config
        user_message = f"""Current configuration:
{json.dumps(current_config, indent=2)}

User request: {prompt}

Return the updated JSON configuration with ALL fields:"""

        # Get model from environment or use default
        model = os.getenv('OLLAMA_MODEL', 'llama3.2')
        
        # Call Ollama
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}
            ],
            options={
                'temperature': 0.3,  # Lower temperature for more consistent outputs
                'num_predict': 500
            }
        )
        
        # Extract response content
        content = response['message']['content'].strip()
        
        # Try to extract JSON from response
        # Remove markdown code blocks if present
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        elif '```' in content:
            content = content.split('```')[1].split('```')[0]
        
        # Find JSON object in response
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            content = content[start:end]
        
        # Parse JSON
        new_config = json.loads(content)
        
        # Validate and sanitize all parameters
        new_config['gravity'] = max(0.1, min(2.0, float(new_config.get('gravity', 0.5))))
        new_config['jump_strength'] = max(-15, min(-5, float(new_config.get('jump_strength', -10))))
        new_config['pipe_speed'] = max(1, min(10, float(new_config.get('pipe_speed', 3))))
        new_config['pipe_gap'] = max(100, min(350, int(new_config.get('pipe_gap', 200))))
        new_config['pipe_frequency'] = max(60, min(150, int(new_config.get('pipe_frequency', 90))))
        
        # Validate colors
        def validate_color(color):
            if isinstance(color, list) and len(color) == 3:
                return [max(0, min(255, int(c))) for c in color]
            return [135, 206, 235]  # Default blue
        
        new_config['background_color'] = validate_color(new_config.get('background_color'))
        new_config['bird_color'] = validate_color(new_config.get('bird_color'))
        new_config['pipe_color'] = validate_color(new_config.get('pipe_color'))
        
        # Validate difficulty
        if new_config.get('difficulty') not in ['easy', 'medium', 'hard']:
            new_config['difficulty'] = 'medium'
        
        return new_config
        
    except ollama.ResponseError as e:
        print(f"⚠️  Ollama API error: {str(e)}")
        print("   Make sure Ollama is running: ollama serve")
        print("   And you have a model installed: ollama pull llama3.2")
        return rule_based_generator(prompt, current_config)
        
    except json.JSONDecodeError as e:
        print(f"⚠️  Could not parse AI response as JSON: {str(e)}")
        print("   Using rule-based fallback...")
        return rule_based_generator(prompt, current_config)
        
    except Exception as e:
        print(f"⚠️  Ollama error: {str(e)}")
        print("   Using rule-based system as fallback...")
        return rule_based_generator(prompt, current_config)

def check_ollama_available():
    """Check if Ollama is installed and running."""
    if not OLLAMA_AVAILABLE:
        return False
    
    try:
        # Try to list models
        ollama.list()
        return True
    except Exception:
        return False

def generate_level(prompt, current_config):
    """
    Main function to generate level configuration from natural language.
    
    Args:
        prompt: User's natural language request
        current_config: Current game configuration dict
        
    Returns:
        Updated configuration dict or None if parsing failed
    """
    
    # Try Ollama first if available
    if check_ollama_available():
        print("🤖 Using Ollama (local AI)")
        result = generate_level_with_ollama(prompt, current_config)
        if result:
            return result
    elif OLLAMA_AVAILABLE:
        print("⚠️  Ollama installed but not running. Start it with: ollama serve")
        print("   Using rule-based system...")
    else:
        print("ℹ️  Ollama not installed. Using rule-based system")
        print("   Install Ollama from: https://ollama.ai")
        print("   Then: pip install ollama && ollama pull llama3.2")
    
    # Fallback to rule-based system
    return rule_based_generator(prompt, current_config)

def test_backends():
    """Test function to check AI backend availability."""
    print("\n" + "="*60)
    print("🔍 AI BACKEND DIAGNOSTICS")
    print("="*60)
    
    print(f"\n1. Ollama package installed: {OLLAMA_AVAILABLE}")
    
    if OLLAMA_AVAILABLE:
        print("2. Checking if Ollama is running...")
        try:
            models = ollama.list()
            print("   ✅ Ollama is running!")
            print(f"   📦 Available models: {len(models.get('models', []))}")
            if models.get('models'):
                for model in models['models'][:5]:  # Show first 5
                    print(f"      - {model['name']}")
        except Exception as e:
            print(f"   ❌ Ollama not running: {e}")
            print("   💡 Start it with: ollama serve")
    else:
        print("   ❌ Install with: pip install ollama")
        print("   💡 Then download a model: ollama pull llama3.2")
    
    print("\n3. Rule-based system: ✅ Always available")
    
    print("\n" + "="*60)
    print("Recommendation:")
    if check_ollama_available():
        print("  🌟 You're all set! Ollama will handle AI requests.")
    else:
        print("  📋 Install Ollama for better AI understanding:")
        print("     1. Visit: https://ollama.ai")
        print("     2. Run: pip install ollama")
        print("     3. Run: ollama pull llama3.2")
        print("     4. Run: ollama serve")
    print("="*60 + "\n")

# Example usage and testing
if __name__ == "__main__":
    print("Testing AI Co-Creator Module\n")
    
    # Run diagnostics
    test_backends()
    
    # Test configuration
    test_config = {
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
    
    # Test prompts
    test_prompts = [
        "make it harder",
        "change background to sunset colors",
        "make the pipes slower",
        "bigger gap between pipes",
        "night mode"
    ]
    
    print("\nTesting with sample prompts:\n")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"{i}. Prompt: '{prompt}'")
        result = generate_level(prompt, test_config)
        if result:
            print(f"   ✅ Success!")
            print(f"   Gravity: {result['gravity']}, Speed: {result['pipe_speed']}, Gap: {result['pipe_gap']}")
            print(f"   Difficulty: {result['difficulty']}")
        else:
            print(f"   ❌ Failed to process")
        print()