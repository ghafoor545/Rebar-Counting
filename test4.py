from ultralytics import YOLO

# Paths
model_path = "/home/nutech/Downloads/best1.pt"
image_path = "/home/nutech/Downloads/image1.jpeg"

# Load model
model = YOLO(model_path)

# Run inference (CPU, no GUI, safe for high counts)
results = model(image_path, device='cpu', imgsz=640, save=False, show=False)

# Count all detected rebars
total_rebars = sum([len(r.boxes) for r in results])
print(f"Detected Rebars: {total_rebars}")

# Optional: save annotated image (lightweight)
from cv2 import imwrite
imwrite("/home/nutech/Downloads/result.jpg", results[0].plot())
print("Annotated image saved at /home/nutech/Downloads/result.jpg")
