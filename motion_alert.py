import os
import logging
import random
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:\\CrimeSense\\motion_alert.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# File paths
BASE_DIR = "C:\\CrimeSense"
VIDEO_PATH = os.path.join(BASE_DIR, "test_trimmed_video.mp4")
ALERT_LOG_PATH = os.path.join(BASE_DIR, "alerts.txt")

# Function to save alert to a file
def save_alert_to_file(alert_message):
    try:
        with open(ALERT_LOG_PATH, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {alert_message}\n")
        logger.info(f"Alert saved to {ALERT_LOG_PATH}")
    except Exception as e:
        logger.error(f"Failed to save alert to {ALERT_LOG_PATH}: {str(e)}")

# Function to simulate motion detection
def detect_motion(video_path):
    logger.info("Starting motion detection simulation...")

    if not os.path.exists(video_path):
        logger.warning(f"Video file {video_path} not found. Proceeding with simulation.")

    is_harmful = True  # Force harmful motion for testing
    logger.info(f"Simulated detection: Is the motion harmful? {is_harmful}")

    if is_harmful:
        alert_message = "Harmful motion detected: Possible weapon in video!"
        logger.info(f"Generated alert: {alert_message}")
        return alert_message
    else:
        logger.info("No harmful motion detected.")
        return None

# Main execution
if __name__ == "__main__":
    alert_message = detect_motion(VIDEO_PATH)
    if alert_message:
        save_alert_to_file(alert_message)
    else:
        logger.info("No harmful motion detected. No alert generated.")