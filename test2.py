from ultralytics import YOLO

# Load model
model_path = "/home/nutech/Downloads/best1.pt"
model = YOLO(model_path)

# Load image
image_path = "/home/nutech/Downloads/image.jpeg"

# Run inference (CPU, no GUI)
results = model(image_path, device='cpu', imgsz=640, save=False, show=False)

# Count all detected rebars
total_rebars = sum([len(r.boxes) for r in results])
print(f"Detected Rebars: {total_rebars}")

# Optional: save annotated image (lightweight)
results[0].plot()  # returns an image with boxes
from cv2 import imwrite
imwrite("/home/nutech/Downloads/result.jpg", results[0].plot())
print("Annotated image saved at /home/nutech/Downloads/result.jpg")
