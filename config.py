from dotenv import load_dotenv
import os

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL")
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN")
CHECK_CAMERAS_INTERVAL = int(os.getenv("CHECK_CAMERAS_INTERVAL", 25))