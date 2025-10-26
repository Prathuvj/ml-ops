import cv2
import mediapipe as mp
import numpy as np
import json
from typing import List, Dict, Optional
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


class PoseDetector:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def detect(self, frame: np.ndarray) -> Optional[Dict[str, Keypoint]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
            
        keypoints = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmark_name = self.mp_pose.PoseLandmark(idx).name
            keypoints[landmark_name] = Keypoint(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=landmark.visibility
            )
        return keypoints
    
    def close(self):
        self.pose.close()


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
        self.frames: List[Frame] = []
        
    def process(self):
        for frame_number, timestamp, frame in self.video_processor.get_frames():
            keypoints = self.pose_detector.detect(frame)
            if keypoints:
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