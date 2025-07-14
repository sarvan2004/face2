import json
import os

def load_config(config_file="recognition_config.json"):
    """Load current configuration"""
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "face_recognition": {
                "min_confidence": 0.5,
                "max_distance": 0.6,
                "min_face_size": 50,
                "consecutive_frames": 3,
                "quality_threshold": 0.3,
                "high_confidence_threshold": 0.6,
                "history_length": 10,
                "consistency_check_frames": 5
            },
            "display": {
                "show_confidence": True,
                "show_distance": True,
                "show_quality": False,
                "tentative_recognition": True
            },
            "logging": {
                "log_level": "INFO",
                "log_errors": True,
                "log_recognition": False
            }
        }

def save_config(config, config_file="recognition_config.json"):
    """Save configuration to file"""
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_file}")

def print_current_settings(config):
    """Print current recognition settings"""
    face_config = config["face_recognition"]
    print("\n" + "="*50)
    print("CURRENT FACE RECOGNITION SETTINGS")
    print("="*50)
    print(f"Min Confidence (YOLO):     {face_config['min_confidence']}")
    print(f"Max Distance (ArcFace):     {face_config['max_distance']}")
    print(f"Min Face Size:              {face_config['min_face_size']} pixels")
    print(f"Quality Threshold:          {face_config['quality_threshold']}")
    print(f"High Confidence Threshold:  {face_config['high_confidence_threshold']}")
    print(f"Consecutive Frames:         {face_config['consecutive_frames']}")
    print(f"History Length:             {face_config['history_length']}")
    print(f"Consistency Check Frames:   {face_config['consistency_check_frames']}")
    print("="*50)

def get_user_input(prompt, current_value, value_type=float, min_val=None, max_val=None):
    """Get user input with validation"""
    while True:
        try:
            user_input = input(f"{prompt} (current: {current_value}): ").strip()
            if user_input == "":
                return current_value
            
            value = value_type(user_input)
            
            if min_val is not None and value < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be <= {max_val}")
                continue
            
            return value
        except ValueError:
            print(f"Please enter a valid {value_type.__name__}")

def adjust_settings():
    """Interactive settings adjustment"""
    config = load_config()
    face_config = config["face_recognition"]
    
    print("\n" + "="*50)
    print("FACE RECOGNITION PARAMETER ADJUSTMENT")
    print("="*50)
    print("Press Enter to keep current value, or enter new value")
    print("="*50)
    
    # Adjust key parameters
    print("\n1. YOLO Detection Settings:")
    face_config["min_confidence"] = get_user_input(
        "Min YOLO confidence (0.1-1.0)", 
        face_config["min_confidence"], 
        float, 0.1, 1.0
    )
    
    face_config["min_face_size"] = get_user_input(
        "Min face size in pixels (20-200)", 
        face_config["min_face_size"], 
        int, 20, 200
    )
    
    print("\n2. ArcFace Recognition Settings:")
    face_config["max_distance"] = get_user_input(
        "Max distance for match (0.1-1.0, lower=stricter)", 
        face_config["max_distance"], 
        float, 0.1, 1.0
    )
    
    face_config["high_confidence_threshold"] = get_user_input(
        "High confidence threshold (0.1-1.0)", 
        face_config["high_confidence_threshold"], 
        float, 0.1, 1.0
    )
    
    print("\n3. Quality and Consistency Settings:")
    face_config["quality_threshold"] = get_user_input(
        "Face quality threshold (0.1-1.0)", 
        face_config["quality_threshold"], 
        float, 0.1, 1.0
    )
    
    face_config["consecutive_frames"] = get_user_input(
        "Consecutive frames for confirmation (1-10)", 
        face_config["consecutive_frames"], 
        int, 1, 10
    )
    
    face_config["consistency_check_frames"] = get_user_input(
        "Frames to check for consistency (2-10)", 
        face_config["consistency_check_frames"], 
        int, 2, 10
    )
    
    face_config["history_length"] = get_user_input(
        "History length to maintain (5-20)", 
        face_config["history_length"], 
        int, 5, 20
    )
    
    # Save the updated configuration
    save_config(config)
    
    print("\n" + "="*50)
    print("UPDATED SETTINGS")
    print("="*50)
    print_current_settings(config)
    
    print("\nRecommendations for reducing misidentification:")
    print("- Lower 'max_distance' for stricter matching")
    print("- Increase 'high_confidence_threshold' for more confident recognition")
    print("- Increase 'consecutive_frames' for more stable recognition")
    print("- Lower 'min_confidence' if faces aren't being detected")
    print("- Increase 'quality_threshold' to ignore poor quality faces")

def show_recommendations():
    """Show recommendations for different scenarios"""
    print("\n" + "="*50)
    print("RECOGNITION PARAMETER RECOMMENDATIONS")
    print("="*50)
    
    print("\n🔴 HIGH MISIDENTIFICATION RATE:")
    print("- Lower max_distance to 0.4-0.5")
    print("- Increase high_confidence_threshold to 0.7-0.8")
    print("- Increase consecutive_frames to 4-5")
    print("- Increase quality_threshold to 0.4-0.5")
    
    print("\n🟡 FACES NOT BEING RECOGNIZED:")
    print("- Increase max_distance to 0.7-0.8")
    print("- Lower high_confidence_threshold to 0.4-0.5")
    print("- Decrease consecutive_frames to 2-3")
    print("- Lower quality_threshold to 0.2-0.3")
    
    print("\n🟢 BALANCED SETTINGS (Recommended):")
    print("- max_distance: 0.6")
    print("- high_confidence_threshold: 0.6")
    print("- consecutive_frames: 3")
    print("- quality_threshold: 0.3")
    print("- min_confidence: 0.5")

def main():
    """Main menu"""
    while True:
        print("\n" + "="*50)
        print("FACE RECOGNITION PARAMETER TOOL")
        print("="*50)
        print("1. View current settings")
        print("2. Adjust settings")
        print("3. Show recommendations")
        print("4. Exit")
        print("="*50)
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            config = load_config()
            print_current_settings(config)
        elif choice == "2":
            adjust_settings()
        elif choice == "3":
            show_recommendations()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main() 