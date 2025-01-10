import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

VEHICLE_CLASSES = [2, 3, 5, 6, 7]  # car(2), motorcycle(3), bus(5), train(6), truck(7)

class VideoTracker:
    def __init__(self):
        self.drawing = False
        self.box_drawn = False
        self.roi_points = []
        self.tracking_area = None
        self.video_path = None
        self.model = YOLO('yolov8n.pt')  # Initialize YOLO model
        self.object_log = {}
        self.current_frame = None
        
    def select_video(self):
        """Open file dialog to select video file"""
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if self.video_path:
            self.draw_roi()
            
    def draw_roi(self):
        """Allow user to draw ROI on first frame"""
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Could not read video file")
            return
            
        self.current_frame = frame.copy()
        cv2.namedWindow('Draw ROI')
        cv2.setMouseCallback('Draw ROI', self.mouse_callback)
        
        while True:
            display_frame = self.current_frame.copy()
            if len(self.roi_points) > 1:
                pts = np.array(self.roi_points, np.int32)
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                
            cv2.imshow('Draw ROI', display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # Press 'c' to confirm ROI
                if len(self.roi_points) > 2:
                    self.tracking_area = np.array(self.roi_points)
                    cv2.destroyWindow('Draw ROI')
                    self.process_video()
                    break
            elif key == ord('r'):  # Press 'r' to reset ROI
                self.roi_points = []
                self.current_frame = frame.copy()
                
        cap.release()
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing ROI"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_points.append([x, y])
            cv2.circle(self.current_frame, (x, y), 5, (0, 255, 0), -1)
            
    def point_inside_polygon(self, point, polygon):
        """Check if a point is inside the drawn polygon using cv2.pointPolygonTest"""
        x, y = point
        return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0
        
    def process_video(self):
        """Process video with YOLO and BoTSORT tracking"""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize tracking dictionary
        tracked_objects = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run YOLO detection with tracking
            results = self.model.track(frame, tracker="botsort.yaml", persist=True, classes=VEHICLE_CLASSES)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                for box, track_id, cls in zip(boxes, track_ids, classes):
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Check if object center is in ROI
                    in_roi = self.point_inside_polygon(
                        (center_x, center_y), 
                        self.tracking_area
                    )
                    
                    track_id = int(track_id)
                    class_name = self.model.names[int(cls)]
                    
                    # Update tracking information
                    if track_id not in tracked_objects:
                        if in_roi:
                            tracked_objects[track_id] = {
                                'class': class_name,
                                'entry_time': datetime.now(),
                                'exit_time': None
                            }
                    else:
                        if not in_roi and tracked_objects[track_id]['exit_time'] is None:
                            tracked_objects[track_id]['exit_time'] = datetime.now()
                    
                    # Draw bounding box and ID
                    color = (0, 255, 0) if in_roi else (0, 0, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{class_name}-{track_id}", 
                              (int(x1), int(y1 - 10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Draw ROI
            cv2.polylines(frame, [self.tracking_area], True, (255, 0, 0), 2)
            
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        # Save tracking log
        self.save_tracking_log(tracked_objects)
        
    def save_tracking_log(self, tracked_objects):
        """Save tracking results to JSON file"""
        log_data = {}
        for track_id, data in tracked_objects.items():
            log_data[str(track_id)] = {
                'class': data['class'],
                'entry_time': data['entry_time'].strftime('%Y-%m-%d %H:%M:%S.%f'),
                'exit_time': data['exit_time'].strftime('%Y-%m-%d %H:%M:%S.%f') 
                    if data['exit_time'] else None
            }
            
        with open('tracking_log.json', 'w') as f:
            json.dump(log_data, f, indent=4)

def main():
    root = tk.Tk()
    root.title("Video Object Tracker")
    
    tracker = VideoTracker()
    
    select_button = tk.Button(
        root, 
        text="Select Video", 
        command=tracker.select_video
    )
    select_button.pack(pady=20)
    
    instructions = tk.Label(
        root,
        text="Instructions:\n"
             "1. Click 'Select Video' to choose a video file\n"
             "2. Click to draw ROI points\n"
             "3. Press 'c' to confirm ROI\n"
             "4. Press 'r' to reset ROI\n"
             "5. Press 'q' to quit processing"
    )
    instructions.pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()