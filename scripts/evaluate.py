"""
Evaluation script for plant disease detection model.
Computes accuracy, precision, recall, F1-score, and confusion matrix.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_model_and_labels(model_path, labels_path, model_type="tensorflow"):
    """Load model and labels based on model type."""
    
    # Load labels
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    if model_type == "tensorflow":
        try:
            import tensorflow as tf
            if model_path.endswith('.json'):
                # Load from JSON + weights
                json_path = model_path
                weights_path = model_path.replace('.json', '.h5')
                
                from tensorflow.keras.models import model_from_json
                with open(json_path, 'r') as json_file:
                    model_json = json_file.read()
                model = model_from_json(model_json)
                model.load_weights(weights_path)
            else:
                model = tf.keras.models.load_model(model_path)
            
            return model, labels
            
        except ImportError:
            print("TensorFlow not available for evaluation")
            return None, None
    
    elif model_type == "pytorch":
        try:
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.load(model_path, map_location=device)
            model.eval()
            return model, labels
            
        except ImportError:
            print("PyTorch not available for evaluation")
            return None, None
    
    elif model_type == "onnx":
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(model_path)
            return session, labels
            
        except ImportError:
            print("ONNX Runtime not available for evaluation")
            return None, None
    
    else:
        print(f"Unsupported model type: {model_type}")
        return None, None


def create_data_generator(test_data_dir, image_size=128, batch_size=32):
    """Create data generator for test dataset."""
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False  # Important for evaluation
        )
        
        return test_generator
        
    except ImportError:
        print("TensorFlow not available for data loading")
        return None


def evaluate_tensorflow_model(model, test_generator, labels):
    """Evaluate TensorFlow model."""
    print("Evaluating TensorFlow model...")
    
    # Get predictions
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = test_generator.classes
    
    return predicted_classes, true_classes, predictions


def calculate_metrics(y_true, y_pred, labels, output_dir):
    """Calculate and save evaluation metrics."""
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=labels, 
        output_dict=True,
        zero_division=0
    )
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Save metrics to JSON
    metrics = {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
    
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    return metrics


def plot_class_performance(metrics, labels, output_dir):
    """Plot per-class performance metrics."""
    
    # Extract per-class metrics
    classes = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for label in labels:
        if label in metrics['classification_report']:
            classes.append(label)
            precision_scores.append(metrics['classification_report'][label]['precision'])
            recall_scores.append(metrics['classification_report'][label]['recall'])
            f1_scores.append(metrics['classification_report'][label]['f1-score'])
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Precision
    axes[0].bar(range(len(classes)), precision_scores, color='skyblue')
    axes[0].set_title('Precision by Class')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Precision')
    axes[0].set_xticks(range(len(classes)))
    axes[0].set_xticklabels(classes, rotation=45, ha='right')
    axes[0].set_ylim(0, 1)
    
    # Recall
    axes[1].bar(range(len(classes)), recall_scores, color='lightgreen')
    axes[1].set_title('Recall by Class')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Recall')
    axes[1].set_xticks(range(len(classes)))
    axes[1].set_xticklabels(classes, rotation=45, ha='right')
    axes[1].set_ylim(0, 1)
    
    # F1-Score
    axes[2].bar(range(len(classes)), f1_scores, color='orange')
    axes[2].set_title('F1-Score by Class')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_xticks(range(len(classes)))
    axes[2].set_xticklabels(classes, rotation=45, ha='right')
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    
    performance_path = output_dir / "class_performance.png"
    plt.savefig(performance_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Class performance plot saved to: {performance_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate plant disease detection model")
    parser.add_argument("--model-path", required=True, help="Path to model file")
    parser.add_argument("--test-data-dir", required=True, help="Path to test dataset directory")
    parser.add_argument("--labels-path", required=True, help="Path to labels JSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory for evaluation results")
    parser.add_argument("--model-type", choices=["tensorflow", "pytorch", "onnx"], 
                       default="tensorflow", help="Model type")
    parser.add_argument("--image-size", type=int, default=128, help="Image size for preprocessing")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.test_data_dir):
        print(f"Test data directory not found: {args.test_data_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.labels_path):
        print(f"Labels file not found: {args.labels_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and labels
    print("Loading model and labels...")
    model, labels = load_model_and_labels(args.model_path, args.labels_path, args.model_type)
    
    if model is None:
        print("Failed to load model")
        sys.exit(1)
    
    print(f"Loaded model with {len(labels)} classes")
    
    # Create test data generator (currently supports TensorFlow only)
    if args.model_type == "tensorflow":
        print("Loading test data...")
        test_generator = create_data_generator(
            args.test_data_dir, 
            args.image_size, 
            args.batch_size
        )
        
        if test_generator is None:
            print("Failed to create test data generator")
            sys.exit(1)
        
        print(f"Test samples: {test_generator.samples}")
        
        # Evaluate model
        y_pred, y_true, predictions = evaluate_tensorflow_model(model, test_generator, labels)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, labels, output_dir)
        
        # Plot class performance
        plot_class_performance(metrics, labels, output_dir)
        
        print(f"\n✓ Evaluation completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    else:
        print(f"Evaluation for {args.model_type} models not yet implemented")
        print("Please implement the evaluation logic for your specific model type")


if __name__ == "__main__":
    main()