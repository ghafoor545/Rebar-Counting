from ultralytics import YOLO
import cv2

# Load your model
model_path = "/home/nutech/Downloads/best (1).pt"
model = YOLO(model_path)

# Load image
image_path = "/home/nutech/Downloads/WhatsApp Image 2025-10-15 at 3.45.56 AM.jpeg"
img = cv2.imread(image_path)

# Run inference
results = model(img)  # or: results = model.predict(img) depending on your ultralytics version

# Show image with detections
results.show()  # opens a window on Pi

# Save results to disk
output_path = "/home/nutech/Downloads/result.jpg"
results.save(output_path)

# Print count of rebars (assuming your model class is 'rebar' and class 0)
rebar_count = sum([len(res.boxes) for res in results])
print(f"Detected Rebars: {rebar_count}")
