import requests
import time
import pandas as pd
import numpy as np
import emotion_model1
from collections import deque
import json
from datetime import datetime

class EnhancedPredictor:
    def __init__(self, model_dir='bio_avatar_models'):
        # Initialize the predictor and load models
        self.predictor = emotion_model1.BiophysiologicalEmotionPredictor()
        self.predictor.load_models(model_dir)
        
        # Store prediction history for accuracy calculation
        self.prediction_history = deque(maxlen=1000)  # Store last 1000 predictions
        self.ground_truth_buffer = deque(maxlen=1000)  # For validation data if available
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        # Store recent predictions for stability analysis
        self.recent_predictions = deque(maxlen=10)
        
    def calculate_prediction_confidence(self, emotion_probs, activity_probs):
        """Calculate confidence scores based on prediction probabilities"""
        
        # Method 1: Maximum probability (most common)
        emotion_max_conf = np.max(emotion_probs)
        activity_max_conf = np.max(activity_probs)
        
        # Method 2: Entropy-based confidence (lower entropy = higher confidence)
        emotion_entropy = -np.sum(emotion_probs * np.log(emotion_probs + 1e-10))
        activity_entropy = -np.sum(activity_probs * np.log(activity_probs + 1e-10))
        
        # Normalize entropy to 0-1 range (lower entropy = higher confidence)
        max_entropy_emotion = np.log(len(self.predictor.emotion_classes))
        max_entropy_activity = np.log(len(self.predictor.activity_classes))
        
        emotion_entropy_conf = 1 - (emotion_entropy / max_entropy_emotion)
        activity_entropy_conf = 1 - (activity_entropy / max_entropy_activity)
        
        # Method 3: Top-2 difference (higher difference = more confident)
        emotion_sorted = np.sort(emotion_probs)[::-1]
        activity_sorted = np.sort(activity_probs)[::-1]
        
        emotion_top2_diff = emotion_sorted[0] - emotion_sorted[1]
        activity_top2_diff = activity_sorted[0] - activity_sorted[1]
        
        confidence_scores = {
            'emotion': {
                'max_prob': float(emotion_max_conf),
                'entropy_based': float(emotion_entropy_conf),
                'top2_difference': float(emotion_top2_diff),
                'combined': float((emotion_max_conf + emotion_entropy_conf + emotion_top2_diff) / 3)
            },
            'activity': {
                'max_prob': float(activity_max_conf),
                'entropy_based': float(activity_entropy_conf),
                'top2_difference': float(activity_top2_diff),
                'combined': float((activity_max_conf + activity_entropy_conf + activity_top2_diff) / 3)
            }
        }
        
        return confidence_scores
    
    def get_confidence_level(self, confidence_score):
        """Convert numerical confidence to categorical level"""
        if confidence_score >= self.confidence_thresholds['high']:
            return 'HIGH'
        elif confidence_score >= self.confidence_thresholds['medium']:
            return 'MEDIUM'
        elif confidence_score >= self.confidence_thresholds['low']:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def calculate_prediction_stability(self, current_emotion, current_activity):
        """Calculate how stable recent predictions are"""
        self.recent_predictions.append((current_emotion, current_activity))
        
        if len(self.recent_predictions) < 3:
            return {'emotion_stability': 1.0, 'activity_stability': 1.0}
        
        recent_emotions = [pred[0] for pred in self.recent_predictions]
        recent_activities = [pred[1] for pred in self.recent_predictions]
        
        # Calculate stability as consistency over recent predictions
        emotion_stability = recent_emotions.count(current_emotion) / len(recent_emotions)
        activity_stability = recent_activities.count(current_activity) / len(recent_activities)
        
        return {
            'emotion_stability': emotion_stability,
            'activity_stability': activity_stability
        }
    
    def calculate_running_accuracy(self, ground_truth_emotion=None, ground_truth_activity=None):
        """Calculate running accuracy if ground truth is available"""
        if not self.prediction_history:
            return None
        
        # This would be used if you have validation data or user feedback
        emotion_correct = 0
        activity_correct = 0
        total_predictions = len(self.prediction_history)
        
        for i, pred in enumerate(self.prediction_history):
            if i < len(self.ground_truth_buffer):
                gt = self.ground_truth_buffer[i]
                if pred['emotion'] == gt.get('emotion'):
                    emotion_correct += 1
                if pred['activity'] == gt.get('activity'):
                    activity_correct += 1
        
        if len(self.ground_truth_buffer) > 0:
            return {
                'emotion_accuracy': emotion_correct / min(total_predictions, len(self.ground_truth_buffer)),
                'activity_accuracy': activity_correct / min(total_predictions, len(self.ground_truth_buffer)),
                'total_evaluated': min(total_predictions, len(self.ground_truth_buffer))
            }
        return None
    
    def enhanced_predict(self, df):
        """Make prediction with confidence and accuracy metrics"""
        try:
            # Preprocess features
            processed_features = self.predictor.preprocess_data(df, fit_transform=False)
            
            if processed_features.shape[0] < self.predictor.sequence_length:
                return None
            
            # Create sequence
            sequence = processed_features[-self.predictor.sequence_length:]
            sequence = sequence.reshape(1, self.predictor.sequence_length, -1)
            
            # Get raw predictions (probabilities)
            emotion_probs, activity_probs = self.predictor.main_model.predict(sequence, verbose=0)
            
            # Get predicted labels
            emotion_idx = emotion_probs.argmax(axis=1)[0]
            activity_idx = activity_probs.argmax(axis=1)[0]
            
            emotion_label = self.predictor.emotion_encoder.inverse_transform([emotion_idx])[0]
            activity_label = self.predictor.activity_encoder.inverse_transform([activity_idx])[0]
            
            # Calculate confidence scores
            confidence_scores = self.calculate_prediction_confidence(
                emotion_probs[0], activity_probs[0]
            )
            
            # Calculate prediction stability
            stability_scores = self.calculate_prediction_stability(emotion_label, activity_label)
            
            # Store prediction in history
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'emotion': emotion_label,
                'activity': activity_label,
                'emotion_confidence': confidence_scores['emotion']['combined'],
                'activity_confidence': confidence_scores['activity']['combined'],
                'emotion_probs': emotion_probs[0].tolist(),
                'activity_probs': activity_probs[0].tolist()
            }
            self.prediction_history.append(prediction_record)
            
            # Calculate running accuracy (if ground truth available)
            running_accuracy = self.calculate_running_accuracy()
            
            # Prepare comprehensive results
            results = {
                'predictions': {
                    'emotion': emotion_label,
                    'activity': activity_label
                },
                'confidence_scores': confidence_scores,
                'confidence_levels': {
                    'emotion': self.get_confidence_level(confidence_scores['emotion']['combined']),
                    'activity': self.get_confidence_level(confidence_scores['activity']['combined'])
                },
                'stability_scores': stability_scores,
                'raw_probabilities': {
                    'emotion': {
                        class_name: float(prob) 
                        for class_name, prob in zip(self.predictor.emotion_classes, emotion_probs[0])
                    },
                    'activity': {
                        class_name: float(prob) 
                        for class_name, prob in zip(self.predictor.activity_classes, activity_probs[0])
                    }
                },
                'running_accuracy': running_accuracy,
                'prediction_count': len(self.prediction_history)
            }
            
            return results
            
        except Exception as e:
            print(f"[Enhanced Prediction Error] {e}")
            return None
    
    def add_ground_truth(self, emotion_truth, activity_truth):
        """Add ground truth for accuracy calculation (for validation/feedback)"""
        self.ground_truth_buffer.append({
            'emotion': emotion_truth,
            'activity': activity_truth,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_model_statistics(self):
        """Get overall model performance statistics"""
        if not self.prediction_history:
            return None
        
        recent_predictions = list(self.prediction_history)[-100:]  # Last 100 predictions
        
        # Average confidence scores
        avg_emotion_conf = np.mean([p['emotion_confidence'] for p in recent_predictions])
        avg_activity_conf = np.mean([p['activity_confidence'] for p in recent_predictions])
        
        # Prediction distribution
        emotion_dist = {}
        activity_dist = {}
        
        for pred in recent_predictions:
            emotion_dist[pred['emotion']] = emotion_dist.get(pred['emotion'], 0) + 1
            activity_dist[pred['activity']] = activity_dist.get(pred['activity'], 0) + 1
        
        return {
            'total_predictions': len(self.prediction_history),
            'recent_predictions_analyzed': len(recent_predictions),
            'average_confidence': {
                'emotion': float(avg_emotion_conf),
                'activity': float(avg_activity_conf)
            },
            'prediction_distribution': {
                'emotion': emotion_dist,
                'activity': activity_dist
            },
            'running_accuracy': self.calculate_running_accuracy()
        }

# Enhanced real-time prediction loop
def run_enhanced_prediction():
    # Initialize enhanced predictor
    enhanced_predictor = EnhancedPredictor('bio_avatar_models')
    
    # Endpoint URL
    DATA_URL = 'https://hareeshkumarm403.pythonanywhere.com/data'
    
    # Store sensor readings
    sensor_window = []
    EXPECTED_KEYS = ['AccelX', 'AccelY', 'AccelZ', 'BPM', 'ECG', 'GyroX', 'GyroY', 'GyroZ', 'IR', 'Red', 'SpO2', 'TempC']
    
    def get_sensor_reading(url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[Error] Status code: {response.status_code}")
                return []
        except Exception as e:
            print(f"[Exception] {e}")
            return []
    
    def clean_sensor_entry(entry):
        try:
            return {k: float(entry[k]) for k in EXPECTED_KEYS if k in entry}
        except Exception as e:
            print(f"Malformed entry skipped: {e}")
            return None
    
    print("Starting enhanced emotion prediction with confidence scoring...")
    print("="*80)
    
    while True:
        latest_data = get_sensor_reading(DATA_URL)
        
        if not latest_data:
            print("No data received.")
            time.sleep(1)
            continue
        
        # Process new data
        for entry in latest_data:
            clean_entry = clean_sensor_entry(entry)
            if clean_entry:
                sensor_window.append(clean_entry)
        
        # Keep only latest 100 samples
        if len(sensor_window) > 100:
            sensor_window = sensor_window[-100:]
        
        if len(sensor_window) < 100:
            print(f"Collecting data: {len(sensor_window)}/100")
            time.sleep(0.5)
            continue
        
        # Convert to DataFrame and make enhanced prediction
        df = pd.DataFrame(sensor_window)
        results = enhanced_predictor.enhanced_predict(df)
        
        if results:
            # Display results
            print(f"\n{'='*60}")
            print(f"PREDICTION RESULTS - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*60}")
            
            print(f" Emotion: {results['predictions']['emotion']} "
                  f"({results['confidence_levels']['emotion']} confidence: "
                  f"{results['confidence_scores']['emotion']['combined']:.3f})")
            
            print(f" Activity: {results['predictions']['activity']} "
                  f"({results['confidence_levels']['activity']} confidence: "
                  f"{results['confidence_scores']['activity']['combined']:.3f})")
            
            print(f"\n CONFIDENCE BREAKDOWN:")
            print(f"   Emotion - Max Prob: {results['confidence_scores']['emotion']['max_prob']:.3f}, "
                  f"Entropy: {results['confidence_scores']['emotion']['entropy_based']:.3f}, "
                  f"Top-2 Diff: {results['confidence_scores']['emotion']['top2_difference']:.3f}")
            
            print(f"   Activity - Max Prob: {results['confidence_scores']['activity']['max_prob']:.3f}, "
                  f"Entropy: {results['confidence_scores']['activity']['entropy_based']:.3f}, "
                  f"Top-2 Diff: {results['confidence_scores']['activity']['top2_difference']:.3f}")
            
            print(f"\n STABILITY:")
            print(f"   Emotion: {results['stability_scores']['emotion_stability']:.3f}")
            print(f"   Activity: {results['stability_scores']['activity_stability']:.3f}")
            
            # Show top 3 probabilities for each prediction
            print(f"\n TOP PROBABILITIES:")
            emotion_probs = results['raw_probabilities']['emotion']
            activity_probs = results['raw_probabilities']['activity']
            
            print("   Emotions:")
            for emotion, prob in sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"     {emotion}: {prob:.3f}")
            
            print("   Activities:")
            for activity, prob in sorted(activity_probs.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"     {activity}: {prob:.3f}")
            
            # Every 10 predictions, show model statistics
            if results['prediction_count'] % 10 == 0:
                stats = enhanced_predictor.get_model_statistics()
                if stats:
                    print(f"\n📋 MODEL STATISTICS:")
                    print(f"   Total Predictions: {stats['total_predictions']}")
                    print(f"   Avg Emotion Confidence: {stats['average_confidence']['emotion']:.3f}")
                    print(f"   Avg Activity Confidence: {stats['average_confidence']['activity']:.3f}")
        
        time.sleep(4)

# Example of how to add ground truth for accuracy calculation
def example_with_ground_truth():
    """Example showing how to use with validation data"""
    enhanced_predictor = EnhancedPredictor('bio_avatar_models')
    
    # Simulate getting a prediction
    # results = enhanced_predictor.enhanced_predict(your_dataframe)
    
    # Add ground truth (e.g., from user feedback or validation set)
    # enhanced_predictor.add_ground_truth('Happy', 'Walking')
    
    # Get updated accuracy
    # accuracy = enhanced_predictor.calculate_running_accuracy()
    # print(f"Current accuracy: {accuracy}")

if __name__ == "__main__":
    run_enhanced_prediction()