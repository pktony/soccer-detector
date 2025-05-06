from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Create a directory to store results if it doesn't exist
results_dir = './conf_threshold_results'
os.makedirs(results_dir, exist_ok=True)

# Load the model
model = YOLO('./train/runs/detect/soccer-v33/weights/best.pt')

# Image path
img_path = './test/test--1.png'

# Create a DataFrame to store metrics
metrics_df = pd.DataFrame(columns=['confidence_threshold', 'num_detections', 'classes_detected'])

# Loop through confidence thresholds from 0 to 1.0 with step 0.05
conf_thresholds = np.arange(0, 1.05, 0.05)

for conf in conf_thresholds:
    print(f"\nProcessing with confidence threshold: {conf:.2f}")
    
    # Run prediction with current confidence threshold
    results = model.predict(source=img_path, conf=conf)
    result = results[0]
    
    # Get number of detections and classes detected
    num_detections = len(result.boxes)
    classes_detected = set(result.boxes.cls.tolist()) if num_detections > 0 else set()
    classes_detected_str = ', '.join([str(int(cls)) for cls in classes_detected])
    
    # Save the result image with confidence threshold in filename
    result_filename = f"{results_dir}/result_conf_{conf:.2f}.jpg"
    result.save(filename=result_filename)
    
    # Add metrics to DataFrame
    metrics_df = pd.concat([metrics_df, pd.DataFrame({
        'confidence_threshold': [f"{conf:.2f}"],
        'num_detections': [num_detections],
        'classes_detected': [classes_detected_str]
    })], ignore_index=True)
    
    print(f"Saved result to {result_filename}")
    print(f"Number of detections: {num_detections}")
    print(f"Classes detected: {classes_detected_str if classes_detected else 'None'}")

# Save metrics to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
metrics_csv_path = f"{results_dir}/metrics_{timestamp}.csv"
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"\nSaved metrics to {metrics_csv_path}")

print("\nConfidence threshold test completed!")
