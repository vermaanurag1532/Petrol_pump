import cv2
import pandas as pd
from datetime import datetime
from collections import deque, defaultdict
import os
import logging
from ultralytics import YOLO
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

class VehicleQueueTracker:
    def __init__(self, video_path, output_excel='vehicle_data.xlsx'):
        self.video_path = video_path
        self.output_excel = output_excel
        self.tracked_vehicles = {}
        self.queue_data = []
        self.vehicle_ids = deque()
        
        # Load YOLOv8 model
        logger.info("Loading YOLOv8 model...")
        self.model = YOLO('yolov8m.pt')
        logger.info("Model loaded successfully!")

        # Define vehicle classes
        self.two_wheelers = {
            3: 'motorcycle',
            4: 'bicycle',
            1: 'scooter'  # Custom class if your model supports it
        }
        
        self.four_wheelers = {
            2: 'car',
            5: 'bus',
            7: 'truck'
        }

        # Define pump position
        self.pumps_positions = [
            (100, 50, 450, 750)
        ]
        
        # Track vehicle positions between frames
        self.last_positions = {}
        self.id_counter = 0
        self.tracking_memory = defaultdict(int)
        
    def get_vehicle_category(self, class_id):
        """Determine if vehicle is 2-wheeler or 4-wheeler"""
        if class_id in self.two_wheelers:
            return '2-wheeler', self.two_wheelers[class_id]
        elif class_id in self.four_wheelers:
            return '4-wheeler', self.four_wheelers[class_id]
        return 'unknown', 'unknown'

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        iou = intersection / float(area1 + area2 - intersection)
        return iou

    def get_vehicle_id(self, current_box):
        """Get vehicle ID based on position overlap with previous frames"""
        max_iou = 0.3  # IOU threshold
        matched_id = None
        
        for vid, last_box in self.last_positions.items():
            iou = self.calculate_iou(current_box, last_box)
            if iou > max_iou:
                max_iou = iou
                matched_id = vid
        
        if matched_id is None:
            self.id_counter += 1
            matched_id = f"V{self.id_counter}"
            
        return matched_id

    def detect_vehicles(self, frame):
        try:
            # Detect all relevant vehicle classes
            results = self.model(frame, classes=list(self.two_wheelers.keys()) + list(self.four_wheelers.keys()), conf=0.3)
            
            boxes = results[0].boxes
            detections = []
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # Adjust confidence threshold based on vehicle type
                if cls in self.two_wheelers:
                    conf_threshold = 0.3  # Lower threshold for two-wheelers
                else:
                    conf_threshold = 0.4  # Normal threshold for four-wheelers
                
                if conf > conf_threshold:
                    category, vehicle_type = self.get_vehicle_category(cls)
                    detections.append({
                        'xmin': x1,
                        'ymin': y1,
                        'xmax': x2,
                        'ymax': y2,
                        'confidence': conf,
                        'class': cls,
                        'category': category,
                        'vehicle_type': vehicle_type
                    })
            
            return pd.DataFrame(detections)
        except Exception as e:
            logger.error(f"Error during vehicle detection: {e}")
            return pd.DataFrame()

    def track_vehicle(self, vehicle, frame, frame_count):
        x_min, y_min = int(vehicle['xmin']), int(vehicle['ymin'])
        x_max, y_max = int(vehicle['xmax']), int(vehicle['ymax'])
        
        # Get stable vehicle ID
        current_box = [x_min, y_min, x_max, y_max]
        vehicle_id = self.get_vehicle_id(current_box)
        self.last_positions[vehicle_id] = current_box
        
        # Initialize tracking data for new vehicles
        if vehicle_id not in self.tracked_vehicles:
            self.tracked_vehicles[vehicle_id] = {
                'entry_time': datetime.now(),
                'exit_time': None,
                'filling_start': None,
                'filling_end': None,
                'filling_pump': None,
                'vehicle_class': vehicle['class'],
                'category': vehicle['category'],
                'vehicle_type': vehicle['vehicle_type']
            }
            logger.info(f"{vehicle['category']} ({vehicle['vehicle_type']}) {vehicle_id} entered at {datetime.now()}")

        # Check pump area
        pump_position = self.pumps_positions[0]
        pump_x_min, pump_y_min, pump_x_max, pump_y_max = pump_position
        
        vehicle_center_x = (x_min + x_max) // 2
        vehicle_center_y = (y_min + y_max) // 2

        if (pump_x_min < vehicle_center_x < pump_x_max and 
            pump_y_min < vehicle_center_y < pump_y_max):
            
            if not self.tracked_vehicles[vehicle_id]['filling_start']:
                self.tracked_vehicles[vehicle_id]['filling_start'] = datetime.now()
                self.tracked_vehicles[vehicle_id]['filling_pump'] = 1
                logger.info(f"{vehicle['category']} {vehicle_id} started filling at {datetime.now()}")
        else:
            if self.tracked_vehicles[vehicle_id]['filling_start'] and not self.tracked_vehicles[vehicle_id]['filling_end']:
                self.tracked_vehicles[vehicle_id]['filling_end'] = datetime.now()
                self.tracked_vehicles[vehicle_id]['exit_time'] = datetime.now()
                logger.info(f"{vehicle['category']} {vehicle_id} exited at {datetime.now()}")

        # Draw bounding box with different colors for different categories
        color = (0, 255, 0) if vehicle['category'] == '2-wheeler' else (255, 0, 0)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Add vehicle type and ID to display
        label = f"{vehicle['vehicle_type']} {vehicle_id}"
        cv2.putText(frame, label, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw pump area
        cv2.rectangle(frame, (pump_x_min, pump_y_min), (pump_x_max, pump_y_max), 
                     (0, 0, 255), 2)
        cv2.putText(frame, "Pump Area", (pump_x_min + 10, pump_y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0

        # Add display counters
        two_wheeler_count = 0
        four_wheeler_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                vehicle_detections = self.detect_vehicles(frame)

                # Update counters
                two_wheeler_count = len([v for v in self.tracked_vehicles.values() 
                                      if v['category'] == '2-wheeler' and not v['exit_time']])
                four_wheeler_count = len([v for v in self.tracked_vehicles.values() 
                                       if v['category'] == '4-wheeler' and not v['exit_time']])

                # Display counters on frame
                cv2.putText(frame, f"2-Wheelers: {two_wheeler_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"4-Wheelers: {four_wheeler_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Clean up old positions periodically
                if frame_count % 10 == 0:
                    self.last_positions = {k: v for k, v in self.last_positions.items() 
                                        if k in self.tracked_vehicles and 
                                        not self.tracked_vehicles[k]['exit_time']}

                # Process detections
                for _, vehicle in vehicle_detections.iterrows():
                    self.track_vehicle(vehicle, frame, frame_count)

                cv2.imshow("Vehicle Queue Tracking", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("Processing interrupted manually.")
        except Exception as e:
            logger.error(f"Error during video processing: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_to_excel()

    def save_to_excel(self):
        for vehicle_id, data in self.tracked_vehicles.items():
            if data['exit_time']:
                filling_time = None
                if data['filling_start'] and data['filling_end']:
                    filling_time = (data['filling_end'] - data['filling_start']).seconds

                self.queue_data.append({
                    'Vehicle ID': vehicle_id,
                    'Category': data['category'],
                    'Vehicle Type': data['vehicle_type'],
                    'Entry Time': data['entry_time'],
                    'Exit Time': data['exit_time'],
                    'Filling Start': data['filling_start'],
                    'Filling End': data['filling_end'],
                    'Filling Duration (Seconds)': filling_time,
                    'Filling Pump': data['filling_pump']
                })

        df = pd.DataFrame(self.queue_data)
        if os.path.exists(self.output_excel):
            existing_df = pd.read_excel(self.output_excel, engine='openpyxl')
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_excel(self.output_excel, index=False, engine='openpyxl')
        logger.info(f"Data saved to {self.output_excel}")

if __name__ == '__main__':
    video_path = 'pump.dav'
    output_excel = 'vehicle_data.xlsx'
    tracker = VehicleQueueTracker(video_path, output_excel)
    tracker.process_video()