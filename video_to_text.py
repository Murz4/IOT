import json
import cv2
from ultralytics import YOLO
import numpy as np
import time
from collections import deque
import logging
import paho.mqtt.client as mqtt


class ObjectDetectionAnalyzer:
    def __init__(self, model_file='yolov8n.pt', min_score=0.5):
        # Initialize the YOLO detector with the specified model
        self.detector = YOLO(model_file)
        self.min_score = min_score
        self.perf_metrics = deque(maxlen=30)  # Store recent performance metrics (FPS)

        # Network connection setup for MQTT
        self.client = mqtt.Client()
        self.client.username_pw_set("test_user", "test_pass")
        self.client.tls_set()  # Use TLS for secure communication
        self.client.connect("u6edac82.ala.us-east-1.emqxsl.com", 8883)
        self.client.loop_start()  # Start MQTT loop

        self.data_channel = "iot/vision/detect"  # MQTT channel for object data
        self.visual_styles = {}  # Store unique colors for object classes

        # Initialize video capture
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.last_update = 0  # Time of the last data update
        self.update_freq = 2.0  # Frequency for sending data updates
        self.prev_data = {}  # Store the last sent data

    def process_frame(self, frame):
        """Analyze the current frame and process detections."""
        start = time.time()

        try:
            # Convert frame to RGB for YOLO processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictions = self.detector(rgb_frame, stream=True)  # Perform detection
            data = self.process_detections(frame, predictions)  # Process detection results

            # Calculate FPS
            rate = 1.0 / max(time.time() - start, 0.001)
            self.perf_metrics.append(rate)

            # Overlay metrics on the frame
            self.display_metrics(frame, data)
            return frame, np.mean(self.perf_metrics), data

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return frame, 0, {}

    def process_detections(self, frame, predictions):
        """Extract and process detection results."""
        data = {}

        for pred in predictions:
            boxes = pred.boxes
            for box in boxes:
                score = float(box.conf[0])
                if score < self.min_score:
                    continue

                # Extract bounding box coordinates and class info
                coords = map(int, box.xyxy[0])
                x1, y1, x2, y2 = coords
                class_id = int(box.cls[0])
                label = pred.names[class_id]

                # Count objects by class
                data[label] = data.get(label, 0) + 1

                # Get visual style for the class
                style = self.get_visual_style(class_id)
                self.draw_bounding_box(frame, x1, y1, x2, y2, label, score, style)

        # Send processed data over MQTT
        self.publish_data(data)
        return data

    def get_visual_style(self, class_id):
        """Assign a unique color to each class."""
        if class_id not in self.visual_styles:
            self.visual_styles[class_id] = tuple(np.random.randint(0, 255, 3).tolist())
        return self.visual_styles[class_id]

    def draw_bounding_box(self, frame, x1, y1, x2, y2, label, score, color):
        """Draw bounding box and label on the frame."""
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {score:.2f}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def display_metrics(self, frame, data):
        """Display FPS and object counts on the frame."""
        y = 30
        cv2.putText(frame, f"FPS: {np.mean(self.perf_metrics):.1f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for name, count in data.items():
            y += 30
            cv2.putText(frame, f"{name}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def publish_data(self, data):
        """Send object data to the MQTT server."""
        now = time.time()
        if now - self.last_update >= self.update_freq or data != self.prev_data:
            if data:
                try:
                    self.client.publish(
                        self.data_channel,
                        json.dumps({"objects": data}),
                        qos=1
                    )
                    self.last_update = now
                    self.prev_data = data.copy()
                except Exception as e:
                    logging.error(f"Error publishing data: {e}")

    def run(self):
        """Main loop for video analysis."""
        try:
            while True:
                success, frame = self.video.read()
                if not success:
                    logging.error("Failed to capture video frame")
                    break

                frame, _, _ = self.process_frame(frame)
                cv2.imshow('Video Analysis', frame)

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.video.release()
            cv2.destroyAllWindows()
            self.client.loop_stop()
            self.client.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = ObjectDetectionAnalyzer()
    analyzer.run()
