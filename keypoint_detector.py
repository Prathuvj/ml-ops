import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import mediapipe as mp
import numpy as np
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class Keypoint:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class Frame:
    frame_number: int
    timestamp: float
    keypoints: Dict[str, Keypoint]


@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)
    
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)
    
    def distance_to(self, other: 'BoundingBox') -> float:
        c1 = self.center()
        c2 = other.center()
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


class PoseDetector:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def detect(self, frame: np.ndarray) -> Optional[Tuple[Dict[str, Keypoint], BoundingBox]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
            
        keypoints = {}
        x_coords = []
        y_coords = []
        
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmark_name = self.mp_pose.PoseLandmark(idx).name
            keypoints[landmark_name] = Keypoint(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=landmark.visibility
            )
            if landmark.visibility > 0.5:
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)
        
        if not x_coords:
            return None
            
        bbox = BoundingBox(
            x_min=min(x_coords),
            y_min=min(y_coords),
            x_max=max(x_coords),
            y_max=max(y_coords)
        )
        
        return keypoints, bbox
    
    def close(self):
        self.pose.close()


class MainDancerTracker:
    def __init__(self, initialization_frames: int = 10):
        self.initialization_frames = initialization_frames
        self.main_dancer_bbox: Optional[BoundingBox] = None
        self.frame_count = 0
        self.candidate_bboxes: List[BoundingBox] = []
        
    def initialize(self, bbox: BoundingBox):
        self.candidate_bboxes.append(bbox)
        
        if len(self.candidate_bboxes) >= self.initialization_frames:
            largest_bbox = max(self.candidate_bboxes, key=lambda b: b.area())
            self.main_dancer_bbox = largest_bbox
            
    def is_initialized(self) -> bool:
        return self.main_dancer_bbox is not None
    
    def is_main_dancer(self, bbox: BoundingBox, max_distance: float = 0.3) -> bool:
        if not self.is_initialized():
            return True
            
        distance = self.main_dancer_bbox.distance_to(bbox)
        area_ratio = bbox.area() / self.main_dancer_bbox.area()
        
        return distance < max_distance and 0.5 < area_ratio < 2.0
    
    def update(self, bbox: BoundingBox):
        if self.is_initialized():
            alpha = 0.7
            self.main_dancer_bbox = BoundingBox(
                x_min=alpha * self.main_dancer_bbox.x_min + (1 - alpha) * bbox.x_min,
                y_min=alpha * self.main_dancer_bbox.y_min + (1 - alpha) * bbox.y_min,
                x_max=alpha * self.main_dancer_bbox.x_max + (1 - alpha) * bbox.x_max,
                y_max=alpha * self.main_dancer_bbox.y_max + (1 - alpha) * bbox.y_max
            )


class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
    def get_frames(self):
        frame_number = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            timestamp = frame_number / self.fps
            yield frame_number, timestamp, frame
            frame_number += 1
            
    def close(self):
        self.cap.release()


class KeypointExtractor:
    def __init__(self, video_path: str, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.video_processor = VideoProcessor(video_path)
        self.pose_detector = PoseDetector(min_detection_confidence, min_tracking_confidence)
        self.tracker = MainDancerTracker()
        self.frames: List[Frame] = []
        
    def process(self):
        for frame_number, timestamp, frame in self.video_processor.get_frames():
            result = self.pose_detector.detect(frame)
            
            if result:
                keypoints, bbox = result
                
                if not self.tracker.is_initialized():
                    self.tracker.initialize(bbox)
                    self.frames.append(Frame(
                        frame_number=frame_number,
                        timestamp=timestamp,
                        keypoints=keypoints
                    ))
                elif self.tracker.is_main_dancer(bbox):
                    self.tracker.update(bbox)
                    self.frames.append(Frame(
                        frame_number=frame_number,
                        timestamp=timestamp,
                        keypoints=keypoints
                    ))
        
        self.video_processor.close()
        self.pose_detector.close()
        
    def to_dict(self) -> Dict:
        return {
            "video_path": self.video_processor.video_path,
            "fps": self.video_processor.fps,
            "total_frames": len(self.frames),
            "frames": [
                {
                    "frame_number": frame.frame_number,
                    "timestamp": frame.timestamp,
                    "keypoints": {k: asdict(v) for k, v in frame.keypoints.items()}
                }
                for frame in self.frames
            ]
        }
    
    def save_json(self, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class DanceKeypointDetector:
    def __init__(self, video_path: str, output_path: str, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.video_path = video_path
        self.output_path = output_path
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
    def run(self):
        extractor = KeypointExtractor(
            self.video_path,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )
        extractor.process()
        extractor.save_json(self.output_path)
        print(f"Keypoints extracted and saved to {self.output_path}")
        print(f"Total frames processed: {len(extractor.frames)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python dance_keypoint_detector.py <video_path> <output_json_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2]
    
    detector = DanceKeypointDetector(video_path, output_path)
    detector.run()