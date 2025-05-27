from .config.env import medrlcot_config
import logging
import os
import re

logger = logging.getLogger("Logger")

def setup_logger():
    # Setup logs
    os.makedirs(medrlcot_config.log_dir, exist_ok=True)
    pattern = re.compile(rf"{re.escape(medrlcot_config.log_name)}(\d+)\.log")

    log_numbers = [0]

    for filename in os.listdir(medrlcot_config.log_dir):
        match = pattern.fullmatch(filename)
        if match:
            log_numbers.append(int(match.group(1)))

    log_number = f'{max(log_numbers)+1:03}'
    log_file = f"{medrlcot_config.log_dir}/{medrlcot_config.log_name}{log_number}.log"
    print(f"Generated new log file {log_file}")

    # File handler (plain)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s || %(levelname)s || %(name)s -  %(message)s"
    ))

    # Stream handler (bolded name)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        f"%(asctime)s || %(levelname)s || %(name)s - %(message)s"
    ))
    
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
    
    logger.info(f"Setup for {medrlcot_config.model_name}'s log done. This is the beginning of the log.")