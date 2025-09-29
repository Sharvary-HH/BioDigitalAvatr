import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import csv
import os

class BiometricDatasetGenerator:
    def __init__(self):
        # Emotion and activity classes
        self.emotion_classes = [
            "Amusement", "Awe", "Enthusiasm", "Liking", "Surprised",
            "Angry", "Disgust", "Fear", "Sad"
        ]
        
        self.activity_classes = ["Standing", "Sitting", "Walking", "Running"]
        
        # Physiological ranges
        self.physiological_ranges = {
            "ECG": [-2.0, 2.0],
            "AccelX": [-20.0, 20.0],
            "AccelY": [-20.0, 20.0],
            "AccelZ": [-20.0, 20.0],
            "GyroX": [-2000.0, 2000.0],
            "GyroY": [-2000.0, 2000.0],
            "GyroZ": [-2000.0, 2000.0],
            "TempC": [35.0, 42.0],
            "IR": [0, 1023],
            "BPM": [50, 200],
            "Red": [5000, 10000],
            "SpO2": [90.0, 100.0]
        }
        
        # Emotion patterns
        self.emotion_patterns = {
            "Amusement": {"BPM_mult": 1.1, "ECG_var": 1.2, "temp_offset": 0.5},
            "Awe": {"BPM_mult": 1.05, "ECG_var": 1.1, "temp_offset": 0.2},
            "Enthusiasm": {"BPM_mult": 1.3, "ECG_var": 1.4, "temp_offset": 1.0},
            "Liking": {"BPM_mult": 1.0, "ECG_var": 1.0, "temp_offset": 0.0},
            "Surprised": {"BPM_mult": 1.4, "ECG_var": 1.5, "temp_offset": 0.8},
            "Angry": {"BPM_mult": 1.5, "ECG_var": 1.6, "temp_offset": 1.2},
            "Disgust": {"BPM_mult": 0.9, "ECG_var": 1.1, "temp_offset": -0.2},
            "Fear": {"BPM_mult": 1.6, "ECG_var": 1.8, "temp_offset": 1.5},
            "Sad": {"BPM_mult": 0.8, "ECG_var": 0.9, "temp_offset": -0.5}
        }
        
        # Activity patterns
        self.activity_patterns = {
            "Standing": {"accel_var": 0.5, "gyro_var": 0.3, "BPM_mult": 1.0},
            "Sitting": {"accel_var": 0.2, "gyro_var": 0.1, "BPM_mult": 0.9},
            "Walking": {"accel_var": 2.0, "gyro_var": 1.5, "BPM_mult": 1.2},
            "Running": {"accel_var": 5.0, "gyro_var": 3.0, "BPM_mult": 1.8}
        }
        
        # Base physiological values
        self.base_values = {
            "ECG": 1.5,
            "AccelX": 9.8,  # Gravity component
            "AccelY": 0.0,
            "AccelZ": 0.0,
            "GyroX": 0.0,
            "GyroY": 0.0,
            "GyroZ": 0.0,
            "TempC": 36.5,
            "IR": 17000,
            "BPM": 70,
            "Red": 7500,
            "SpO2": 98.0
        }
    
    def generate_timestamp_sequence(self, start_time, duration_minutes, frequency_hz=1):
        """Generate timestamp sequence"""
        timestamps = []
        current_time = datetime.fromisoformat(start_time.replace('Z', '+00:00')) if isinstance(start_time, str) else start_time
        
        total_samples = int(duration_minutes * 60 * frequency_hz)
        time_delta = timedelta(seconds=1/frequency_hz)
        
        for i in range(total_samples):
            timestamps.append(current_time.strftime("%Y-%m-%dT%H:%M:%S.%f"))
            current_time += time_delta
            
        return timestamps
    
    def add_noise(self, value, noise_factor=0.1):
        """Add realistic noise to sensor values"""
        noise = np.random.normal(0, abs(value) * noise_factor)
        return value + noise
    
    def generate_emotion_influenced_data(self, emotion, activity, base_values, duration_seconds=60):
        """Generate data influenced by emotion and activity patterns"""
        data_points = []
        
        # Get emotion and activity patterns
        emotion_pattern = self.emotion_patterns[emotion]
        activity_pattern = self.activity_patterns[activity]
        
        # Calculate combined multipliers
        bpm_mult = emotion_pattern["BPM_mult"] * activity_pattern["BPM_mult"]
        temp_offset = emotion_pattern["temp_offset"]
        ecg_var = emotion_pattern["ECG_var"]
        accel_var = activity_pattern["accel_var"]
        gyro_var = activity_pattern["gyro_var"]
        
        for i in range(duration_seconds):
            data_point = {}
            
            # Generate BPM with emotion and activity influence
            base_bpm = base_values["BPM"]
            bpm = base_bpm * bpm_mult + np.random.normal(0, 5)
            bpm = np.clip(bpm, self.physiological_ranges["BPM"][0], self.physiological_ranges["BPM"][1])
            data_point["BPM"] = round(bpm)
            
            # Generate ECG with emotion influence
            base_ecg = base_values["ECG"]
            ecg = base_ecg + np.random.normal(0, 0.3 * ecg_var)
            ecg = np.clip(ecg, self.physiological_ranges["ECG"][0], self.physiological_ranges["ECG"][1])
            data_point["ECG"] = round(ecg, 3)
            
            # Generate temperature with emotion influence
            temp = base_values["TempC"] + temp_offset + np.random.normal(0, 0.2)
            temp = np.clip(temp, self.physiological_ranges["TempC"][0], self.physiological_ranges["TempC"][1])
            data_point["TempC"] = round(temp, 1)
            
            # Generate accelerometer data with activity influence
            if activity == "Walking":
                # Walking pattern - sinusoidal movement
                t = i * 0.1
                accel_x = base_values["AccelX"] + 2 * np.sin(t * 2) * accel_var + np.random.normal(0, 0.5)
                accel_y = base_values["AccelY"] + 1.5 * np.cos(t * 2.5) * accel_var + np.random.normal(0, 0.3)
                accel_z = base_values["AccelZ"] + 1 * np.sin(t * 1.5) * accel_var + np.random.normal(0, 0.2)
            elif activity == "Running":
                # Running pattern - higher amplitude, faster frequency
                t = i * 0.1
                accel_x = base_values["AccelX"] + 4 * np.sin(t * 4) * accel_var + np.random.normal(0, 1.0)
                accel_y = base_values["AccelY"] + 3 * np.cos(t * 4.5) * accel_var + np.random.normal(0, 0.8)
                accel_z = base_values["AccelZ"] + 2 * np.sin(t * 3) * accel_var + np.random.normal(0, 0.5)
            else:
                # Standing/Sitting - minimal movement
                accel_x = base_values["AccelX"] + np.random.normal(0, 0.1 * accel_var)
                accel_y = base_values["AccelY"] + np.random.normal(0, 0.1 * accel_var)
                accel_z = base_values["AccelZ"] + np.random.normal(0, 0.1 * accel_var)
            
            data_point["AccelX"] = round(np.clip(accel_x, *self.physiological_ranges["AccelX"]), 3)
            data_point["AccelY"] = round(np.clip(accel_y, *self.physiological_ranges["AccelY"]), 3)
            data_point["AccelZ"] = round(np.clip(accel_z, *self.physiological_ranges["AccelZ"]), 3)
            
            # Generate gyroscope data with activity influence
            if activity in ["Walking", "Running"]:
                gyro_mult = 2 if activity == "Running" else 1
                gyro_x = np.random.normal(0, 10 * gyro_var * gyro_mult)
                gyro_y = np.random.normal(0, 8 * gyro_var * gyro_mult)
                gyro_z = np.random.normal(0, 5 * gyro_var * gyro_mult)
            else:
                gyro_x = np.random.normal(0, 1 * gyro_var)
                gyro_y = np.random.normal(0, 1 * gyro_var)
                gyro_z = np.random.normal(0, 1 * gyro_var)
            
            data_point["GyroX"] = round(np.clip(gyro_x, *self.physiological_ranges["GyroX"]), 3)
            data_point["GyroY"] = round(np.clip(gyro_y, *self.physiological_ranges["GyroY"]), 3)
            data_point["GyroZ"] = round(np.clip(gyro_z, *self.physiological_ranges["GyroZ"]), 3)
            
            # Generate IR and Red values (correlated with BPM and SpO2)
            ir_base = base_values["IR"]
            ir = ir_base + np.random.normal(0, 500) + (bpm - 70) * 10
            data_point["IR"] = int(np.clip(ir, *self.physiological_ranges["IR"]))
            
            red_base = base_values["Red"]
            red = red_base + np.random.normal(0, 300) + (bpm - 70) * 5
            data_point["Red"] = int(np.clip(red, *self.physiological_ranges["Red"]))
            
            # Generate SpO2 (slightly affected by activity)
            spo2_base = base_values["SpO2"]
            if activity == "Running":
                spo2 = spo2_base - 2 + np.random.normal(0, 1)
            else:
                spo2 = spo2_base + np.random.normal(0, 0.5)
            data_point["SpO2"] = round(np.clip(spo2, *self.physiological_ranges["SpO2"]), 1)
            
            data_points.append(data_point)
        
        return data_points
    
    def generate_session_data(self, session_duration_minutes=10, emotion_change_interval=2):
        """Generate a complete session with emotion and activity changes"""
        session_data = []
        timestamps = self.generate_timestamp_sequence(
            datetime.now(), 
            session_duration_minutes, 
            frequency_hz=1
        )
        
        current_emotion = random.choice(self.emotion_classes)
        current_activity = random.choice(self.activity_classes)
        
        data_points = self.generate_emotion_influenced_data(
            current_emotion, 
            current_activity, 
            self.base_values, 
            duration_seconds=len(timestamps)
        )
        
        # Add random emotion/activity changes
        emotion_change_points = random.sample(
            range(len(timestamps)), 
            min(3, len(timestamps) // (emotion_change_interval * 60))
        )
        
        for i, timestamp in enumerate(timestamps):
            if i in emotion_change_points:
                current_emotion = random.choice(self.emotion_classes)
                current_activity = random.choice(self.activity_classes)
            
            row = {
                "Timestamp": timestamp,
                **data_points[i],
                "emotion_label": current_emotion,
                "activity_label": current_activity
            }
            session_data.append(row)
        
        return session_data
    
    def generate_dataset(self, 
                        num_sessions=50, 
                        session_duration_minutes=10, 
                        output_file="biometric_dataset.csv"):
        """Generate complete dataset with multiple sessions"""
        
        print(f"Generating dataset with {num_sessions} sessions...")
        all_data = []
        
        for session in range(num_sessions):
            if session % 10 == 0:
                print(f"Generating session {session + 1}/{num_sessions}")
            
            # Vary session start time
            start_time = datetime.now() + timedelta(days=session, hours=random.randint(8, 20))
            
            session_data = []
            timestamps = self.generate_timestamp_sequence(
                start_time, 
                session_duration_minutes, 
                frequency_hz=1
            )
            
            # Generate realistic emotion/activity sequences for each session
            emotion_sequence = self.generate_realistic_emotion_sequence(len(timestamps))
            activity_sequence = self.generate_realistic_activity_sequence(len(timestamps))
            
            for i, timestamp in enumerate(timestamps):
                emotion = emotion_sequence[i]
                activity = activity_sequence[i]
                
                # Generate single data point
                data_point = self.generate_single_datapoint(emotion, activity)
                
                row = {
                    "Timestamp": timestamp,
                    **data_point,
                    "emotion_label": emotion,
                    "activity_label": activity
                }
                all_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(all_data)
        
        # Reorder columns to match your specification
        column_order = [
            "Timestamp", "ECG", "AccelX", "AccelY", "AccelZ", 
            "GyroX", "GyroY", "GyroZ", "TempC", "IR", "BPM", 
            "Red", "SpO2", "emotion_label", "activity_label"
        ]
        df = df[column_order]
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Dataset saved to {output_file}")
        print(f"Total samples: {len(df)}")
        print(f"Emotion distribution:\n{df['emotion_label'].value_counts()}")
        print(f"Activity distribution:\n{df['activity_label'].value_counts()}")
        
        return df
    
    def generate_realistic_emotion_sequence(self, length):
        """Generate realistic emotion sequence with transitions"""
        emotions = []
        current_emotion = random.choice(self.emotion_classes)
        emotion_duration = random.randint(30, 120)  # 30-120 seconds per emotion
        
        for i in range(length):
            if i % emotion_duration == 0 and i > 0:
                # Change emotion with some probability
                if random.random() < 0.7:  # 70% chance to change
                    current_emotion = random.choice(self.emotion_classes)
                emotion_duration = random.randint(30, 120)
            
            emotions.append(current_emotion)
        
        return emotions
    
    def generate_realistic_activity_sequence(self, length):
        """Generate realistic activity sequence"""
        activities = []
        current_activity = random.choice(self.activity_classes)
        activity_duration = random.randint(60, 300)  # 1-5 minutes per activity
        
        for i in range(length):
            if i % activity_duration == 0 and i > 0:
                # Activity transitions are more structured
                transition_prob = {
                    "Sitting": {"Standing": 0.4, "Walking": 0.3, "Sitting": 0.3},
                    "Standing": {"Sitting": 0.3, "Walking": 0.4, "Running": 0.1, "Standing": 0.2},
                    "Walking": {"Standing": 0.3, "Running": 0.2, "Sitting": 0.2, "Walking": 0.3},
                    "Running": {"Walking": 0.6, "Standing": 0.3, "Running": 0.1}
                }
                
                probs = transition_prob.get(current_activity, {})
                if probs:
                    current_activity = np.random.choice(
                        list(probs.keys()), 
                        p=list(probs.values())
                    )
                
                activity_duration = random.randint(60, 300)
            
            activities.append(current_activity)
        
        return activities
    
    def generate_single_datapoint(self, emotion, activity):
        """Generate a single data point based on emotion and activity"""
        emotion_pattern = self.emotion_patterns[emotion]
        activity_pattern = self.activity_patterns[activity]
        
        # Calculate multipliers
        bpm_mult = emotion_pattern["BPM_mult"] * activity_pattern["BPM_mult"]
        temp_offset = emotion_pattern["temp_offset"]
        ecg_var = emotion_pattern["ECG_var"]
        accel_var = activity_pattern["accel_var"]
        gyro_var = activity_pattern["gyro_var"]
        
        # Generate values
        bpm = self.base_values["BPM"] * bpm_mult + np.random.normal(0, 5)
        bpm = np.clip(bpm, *self.physiological_ranges["BPM"])
        
        ecg = self.base_values["ECG"] + np.random.normal(0, 0.3 * ecg_var)
        ecg = np.clip(ecg, *self.physiological_ranges["ECG"])
        
        temp = self.base_values["TempC"] + temp_offset + np.random.normal(0, 0.2)
        temp = np.clip(temp, *self.physiological_ranges["TempC"])
        
        # Accelerometer based on activity
        if activity == "Walking":
            accel_noise = 0.5
        elif activity == "Running":
            accel_noise = 1.0
        else:
            accel_noise = 0.1
        
        accel_x = self.base_values["AccelX"] + np.random.normal(0, accel_noise * accel_var)
        accel_y = self.base_values["AccelY"] + np.random.normal(0, accel_noise * accel_var)
        accel_z = self.base_values["AccelZ"] + np.random.normal(0, accel_noise * accel_var)
        
        # Gyroscope
        gyro_noise = 5 if activity in ["Walking", "Running"] else 1
        gyro_x = np.random.normal(0, gyro_noise * gyro_var)
        gyro_y = np.random.normal(0, gyro_noise * gyro_var)
        gyro_z = np.random.normal(0, gyro_noise * gyro_var)
        
        # IR and Red
        ir = self.base_values["IR"] + np.random.normal(0, 500) + (bpm - 70) * 10
        red = self.base_values["Red"] + np.random.normal(0, 300) + (bpm - 70) * 5
        
        # SpO2
        spo2 = self.base_values["SpO2"] + np.random.normal(0, 0.5)
        if activity == "Running":
            spo2 -= 2
        
        return {
            "ECG": round(ecg, 3),
            "AccelX": round(np.clip(accel_x, *self.physiological_ranges["AccelX"]), 3),
            "AccelY": round(np.clip(accel_y, *self.physiological_ranges["AccelY"]), 3),
            "AccelZ": round(np.clip(accel_z, *self.physiological_ranges["AccelZ"]), 3),
            "GyroX": round(np.clip(gyro_x, *self.physiological_ranges["GyroX"]), 3),
            "GyroY": round(np.clip(gyro_y, *self.physiological_ranges["GyroY"]), 3),
            "GyroZ": round(np.clip(gyro_z, *self.physiological_ranges["GyroZ"]), 3),
            "TempC": round(temp, 1),
            "IR": int(np.clip(ir, *self.physiological_ranges["IR"])),
            "BPM": int(bpm),
            "Red": int(np.clip(red, *self.physiological_ranges["Red"])),
            "SpO2": round(np.clip(spo2, *self.physiological_ranges["SpO2"]), 1)
        }

# Usage example
if __name__ == "__main__":
    # Create generator instance
    generator = BiometricDatasetGenerator()
    
    # Generate dataset
    # Parameters you can adjust:
    # - num_sessions: Number of different recording sessions (default: 50)
    # - session_duration_minutes: Duration of each session (default: 10 minutes)
    # - output_file: Name of the output CSV file
    
    dataset = generator.generate_dataset(
        num_sessions=100,           # Generate 100 sessions
        session_duration_minutes=5, # Each session 5 minutes long
        output_file="biometric_training_dataset.csv"
    )
    
    print("\nDataset generation complete!")
    print("Sample data:")
    print(dataset.head())
    
    # Generate a smaller test dataset
    test_dataset = generator.generate_dataset(
        num_sessions=20,
        session_duration_minutes=3,
        output_file="biometric_test_dataset.csv"
    )
    
    print("\nTest dataset generation complete!")