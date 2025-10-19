"""
Standalone prediction script for financial complaint classification.
"""
import torch
import pickle
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ComplaintClassifier:
    def __init__(self, model_path="models/distilbert-model"):
        """Initialize the classifier."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mapping
        with open(os.path.join(model_path, "label_mapping.pkl"), "rb") as f:
            label_mapping = pickle.load(f)
        self.label_names = label_mapping["label_names"]
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Categories: {self.label_names}")
    
    def predict(self, text, max_length=128):
        """
        Predict the category of a financial complaint.
        
        Args:
            text (str): The complaint text
            max_length (int): Maximum sequence length
            
        Returns:
            dict: Prediction results with probabilities
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get probabilities
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Format results
        results = {
            "predicted_class": predicted_class,
            "predicted_label": self.label_names[predicted_class],
            "confidence": confidence,
            "all_probabilities": {
                self.label_names[i]: probabilities[0][i].item()
                for i in range(len(self.label_names))
            }
        }
        
        return results


def main():
    """Example usage."""
    # Initialize classifier
    classifier = ComplaintClassifier()
    
    # Example complaints
    test_complaints = [
        "There are incorrect items on my credit report that do not belong to me.",
        "I am being harassed by debt collectors calling multiple times per day.",
        "My personal loan interest rate is much higher than promised.",
        "My mortgage payment increased without proper notification."
    ]
    
    print("\n" + "="*80)
    print("FINANCIAL COMPLAINT CLASSIFICATION - PREDICTIONS")
    print("="*80 + "\n")
    
    # Predict for each complaint
    for i, complaint in enumerate(test_complaints, 1):
        print(f"Test Case {i}:")
        print(f"Text: {complaint}")
        print("-" * 80)
        
        results = classifier.predict(complaint)
        
        print(f"Predicted Category: {results['predicted_label']}")
        print(f"Confidence: {results['confidence']:.4f}")
        print("\nAll Probabilities:")
        for label, prob in results['all_probabilities'].items():
            print(f"  {label}: {prob:.4f}")
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
