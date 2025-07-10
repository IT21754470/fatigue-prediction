"""
Swimming recommendation service with ML model loading and prediction logic
"""
import pickle
import pandas as pd
import numpy as np
from utils.recommendation_utils import get_recommendation_text

class RecommendationService:
    def __init__(self):
        self.model = None
        self.load_models()
    
    def load_models(self):
        """Load model components when the service starts"""
        print("Loading model components...")
        try:
            # Try to load the grshort model
            with open('grshort_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            print("✅ Grshort model loaded successfully!")
        except FileNotFoundError:
            print("❌ grshort_model.pkl not found")
            self.model = None
        except Exception as e:
            print(f"❌ Error loading grshort model: {e}")
            self.model = None
    
    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.model is not None
    
    def get_available_codes(self):
        """Get all available grshort codes"""
        if self.model is not None and hasattr(self.model, 'classes_'):
            return sorted(self.model.classes_.tolist())
        # Return default codes if model doesn't have classes_
        from utils.recommendation_utils import RECOMMENDATIONS
        return sorted(list(RECOMMENDATIONS.keys()))
    
    def predict_grshort(self, improvement, fatigue_level, stroke_type):
        """Predict grshort code for swimmer"""
        if self.model is None:
            return "ERROR", 0.0, []
        
        try:
            # Create feature vector based on available inputs
            # Since the actual model has different features, we'll need to adapt
            features = pd.DataFrame({
                'predicted improvement (s)': [improvement],
                'Predicted Fatigue Level': [fatigue_level],
                'Stroke Type': [stroke_type],
                'is_positive_improvement': [improvement > 0],
                'improvement_magnitude': [abs(improvement)]
            })

            # If the model is a pipeline, it should handle preprocessing
            if hasattr(self.model, 'predict'):
                grshort_code = self.model.predict(features)[0]
                
                # Try to get probabilities if available
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(features)[0]
                    confidence = max(probabilities)
                    
                    # Get top 3 predictions if classes are available
                    if hasattr(self.model, 'classes_'):
                        class_probs = dict(zip(self.model.classes_, probabilities))
                        top_predictions = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    else:
                        top_predictions = [(grshort_code, confidence)]
                else:
                    confidence = 1.0  # Default confidence
                    top_predictions = [(grshort_code, 1.0)]
                
                return grshort_code, confidence, top_predictions
            else:
                return "ERROR", 0.0, []
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return a default recommendation based on simple logic
            default_code = self._get_default_recommendation(improvement, fatigue_level)
            return default_code, 0.5, [(default_code, 0.5)]
    
    def _get_default_recommendation(self, improvement, fatigue_level):
        """Provide default recommendation when model fails"""
        if improvement > 0:
            return "MC"  # Maintain current training
        elif fatigue_level.lower() in ['high', 'very high']:
            return "RDC"  # Reduce training intensity
        else:
            return "FOT"  # Focus on technique

# Global service instance
recommendation_service = RecommendationService()