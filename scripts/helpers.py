import os
import logging

logger = logging.getLogger('ml_pipeline')  # Set the logger name
handler = logging.FileHandler('/app/ml_pipeline.log')
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def create_directories(directories):
    for label, path in directories.items():
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory for {label}: {path}")
        else:
            logger.info(f"Directory for {label} already exists: {path}")