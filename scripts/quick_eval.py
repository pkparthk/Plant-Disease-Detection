"""
Quick model evaluation script to test accuracy on sample images.
"""

import os
import json
import numpy as np
from tensorflow import keras
from PIL import Image
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

def load_test_images(test_dir, labels):
    """Load and preprocess test images."""
    images = []
    true_labels = []
    
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                # Load and preprocess image
                img_path = os.path.join(test_dir, filename)
                img = Image.open(img_path)
                img = img.resize((128, 128))
                img = img.convert('RGB')
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
                
                # Try to infer label from filename
                predicted_label = None
                for label in labels:
                    if any(part in filename.lower() for part in label.lower().split('_')):
                        predicted_label = label
                        break
                
                if predicted_label:
                    true_labels.append(predicted_label)
                else:
                    # Default for testing
                    true_labels.append("unknown")
                    
                print(f"Loaded: {filename} -> {predicted_label}")
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return np.array(images), true_labels

def evaluate_model():
    """Evaluate the current model."""
    try:
        # Load model
        model_path = "../backend/models/plant_disease_model.keras"
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully: {model.input_shape}, {model.output_shape}")
        
        # Load labels
        labels_path = "../backend/labels.json"
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        print(f"Labels loaded: {len(labels)} classes")
        
        # Load test images
        test_dir = "../ML_Model/test_set"
        if not os.path.exists(test_dir):
            print(f"Test directory not found: {test_dir}")
            return
            
        test_images, true_labels = load_test_images(test_dir, labels)
        print(f"Loaded {len(test_images)} test images")
        
        if len(test_images) == 0:
            print("No test images found!")
            return
        
        # Make predictions
        predictions = model.predict(test_images)
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        # Show results
        print("\n=== PREDICTION RESULTS ===")
        correct = 0
        for i, (true_label, pred_idx, confidence) in enumerate(zip(true_labels, predicted_classes, confidence_scores)):
            predicted_label = labels[pred_idx] if pred_idx < len(labels) else "unknown"
            is_correct = true_label.lower() in predicted_label.lower() or predicted_label.lower() in true_label.lower()
            if is_correct:
                correct += 1
            
            print(f"Image {i+1}:")
            print(f"  True: {true_label}")
            print(f"  Predicted: {predicted_label} ({confidence:.2%})")
            print(f"  Correct: {is_correct}")
            print()
        
        accuracy = correct / len(test_images) if len(test_images) > 0 else 0
        print(f"Overall Accuracy: {accuracy:.2%} ({correct}/{len(test_images)})")
        
        # Show confidence distribution
        print(f"\nConfidence Statistics:")
        print(f"  Mean: {np.mean(confidence_scores):.2%}")
        print(f"  Min: {np.min(confidence_scores):.2%}")
        print(f"  Max: {np.max(confidence_scores):.2%}")
        
        return accuracy, confidence_scores
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    evaluate_model()