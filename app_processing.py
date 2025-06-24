import cv2
import os
import logging
import json
from ultralytics import YOLO

# Set up logging to both console and file for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Directories and file paths
OUTPUT_DIR = "outputs"
STATIC_DIR = "static"
VIDEO_PATH = "test_trimmed_video.mp4"  # Single video in the project directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Process the single video
def process_video(video_path, output_dir, static_dir):
    logger.info("Starting video processing for object detection using YOLOv5...")
    summaries = {}

    # Check if the video file exists
    if not os.path.exists(video_path):
        logger.error(f"Video file {video_path} not found.")
        return summaries

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_video_path = os.path.join(output_dir, f'temp_{video_name}.mp4')
    output_video_path = os.path.join(output_dir, f'output_{video_name}.mp4')
    converted_video_path = os.path.join(static_dir, f'converted_{video_name}.mp4')
    sample_frame_path = os.path.join(output_dir, 'sample_frame.jpg')

    # Standardize video (6 seconds, 640x360, 30fps)
    cmd = f"ffmpeg -i {video_path} -ss 00:00:00 -t 00:00:06 -vf scale=640:360 -r 30 -c:v libx264 -c:a aac -f mp4 -y {temp_video_path}"
    logger.info(f"Running FFmpeg command to standardize video: {cmd}")
    if os.system(cmd) != 0 or not os.path.exists(temp_video_path):
        logger.error(f"Failed to standardize video {video_path}")
        return summaries
    logger.info(f"Standardized video saved to {temp_video_path}")

    # Process video
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video {temp_video_path}")
        return summaries

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video {video_name}: width={width}, height={height}, fps={fps}, total_frames={total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        logger.error(f"Could not initialize video writer for {output_video_path}")
        cap.release()
        return summaries
    logger.info(f"Video writer initialized for {output_video_path}")

    # Load YOLOv5 model
    logger.info("Loading YOLOv5 model...")
    try:
        model = YOLO("yolov5mu.pt")  # Ensure yolov5mu.pt is in C:/CrimeSense/
        model = model.to('cpu')
        logger.info(f"Model loaded successfully. Class names: {model.names}")
    except Exception as e:
        logger.error(f"Failed to load YOLOv5 model: {str(e)}")
        cap.release()
        return summaries

    WEAPON_CLASSES = {43: "gun", 44: "knife", 76: "scissors"}  # COCO indices for weapons
    PERSON_CLASS = 0  # COCO index for person
    alerts = []
    detection_counts = {"person": 0, "gun": 0, "knife": 0, "scissors": 0}
    frame_count = 0
    sample_frame_written = False

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            logger.info(f"End of video {video_name} reached at frame {frame_count}")
            break

        logger.info(f"Processing frame {frame_count + 1}/{total_frames}")
        try:
            # Run YOLOv5 detection
            results = model(frame)
            detections_found = False
            if results and len(results) > 0 and results[0].boxes:
                logger.info(f"Frame {frame_count + 1}: YOLOv5 detected {len(results[0].boxes)} objects")
                for detection in results[0].boxes:
                    class_id = int(detection.cls)
                    confidence = float(detection.conf)
                    class_name = model.names[class_id] if class_id < len(model.names) else "Unknown"
                    logger.info(f"Frame {frame_count + 1}: Detected class_id={class_id}, class_name={class_name}, confidence={confidence:.2f}")

                    # Detect persons
                    if class_id == PERSON_CLASS and confidence > 0.2:
                        x1, y1, x2, y2 = map(int, detection.xyxy[0])
                        label = f"Person: {confidence:.2f}"
                        logger.info(f"Drawing person box at ({x1}, {y1}, {x2}, {y2}) with label '{label}'")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        detection_counts["person"] += 1
                        detections_found = True

                    # Detect weapons
                    elif class_id in WEAPON_CLASSES and confidence > 0.2:
                        x1, y1, x2, y2 = map(int, detection.xyxy[0])
                        class_name = WEAPON_CLASSES[class_id]
                        label = f"{class_name}: {confidence:.2f}"
                        logger.info(f"Drawing weapon box ({class_name}) at ({x1}, {y1}, {x2}, {y2}) with label '{label}'")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        detection_counts[class_name] += 1
                        alerts.append(f"Weapon ({class_name}) detected in frame {frame_count + 1}: Confidence {confidence:.2f}")
                        detections_found = True

                # Save a sample frame with detections for debugging (only once)
                if detections_found and not sample_frame_written and frame_count > 10:
                    cv2.imwrite(sample_frame_path, frame)
                    logger.info(f"Sample frame with detections saved to {sample_frame_path}")
                    sample_frame_written = True

            else:
                logger.warning(f"Frame {frame_count + 1}: No objects detected by YOLOv5")

        except Exception as e:
            logger.error(f"Error processing frame {frame_count + 1}: {str(e)}")

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    logger.info(f"Processed video saved to {output_video_path}")

    # Convert video to H.264 for web compatibility
    cmd = f"ffmpeg -i {output_video_path} -c:v libx264 -c:a aac -f mp4 -y {converted_video_path}"
    logger.info(f"Running FFmpeg command to convert video: {cmd}")
    if os.system(cmd) != 0:
        logger.warning(f"Failed to convert video {output_video_path}")
    else:
        logger.info(f"Converted video saved to {converted_video_path}")

    # Summarize results
    summaries[video_name] = {
        "total_frames": frame_count,
        "detections": detection_counts,
    }

    with open("summaries.json", "w") as f:
        json.dump(summaries, f)
    with open("alerts.json", "w") as f:
        json.dump(alerts, f)

    logger.info("Video processing completed successfully!")
    return summaries

# Main execution
if __name__ == "__main__":
    summaries = process_video(VIDEO_PATH, OUTPUT_DIR, STATIC_DIR)