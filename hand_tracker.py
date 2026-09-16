import cv2
import torch
from ultralytics import YOLO

# Select compute device dynamically (CUDA if available, else CPU)
device = 0 if torch.cuda.is_available() else "cpu"

# Load custom trained YOLO model
model = YOLO("hand.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("[-] Error: Unable to access camera feed.")
    exit()

print(f"[+] Model running on device: {device}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Perform inference
    results = model.predict(source=frame, device=device, conf=0.5, verbose=False)

    for r in results:
        for box in r.boxes:
            # Extract bounding box and centroid in xywh format
            cx, cy, w, h = box.xywh[0].cpu().numpy()
            center_x, center_y = int(cx), int(cy)
            conf = float(box.conf[0].cpu().numpy())

            # Draw centroid marker
            cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)

            # Display pixel coordinates and confidence
            label = f"Center: ({center_x}, {center_y}) | Conf: {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (center_x + 10, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Real-Time Hand Centroid Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()