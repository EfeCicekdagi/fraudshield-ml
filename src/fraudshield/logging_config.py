import logging
import sys

def setup_logger(name: str = "fraudshield") -> logging.Logger:
    """Setup and return a standard logger for the project."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # Prevent double logging if root logger is also configured
        logger.propagate = False
        
    return logger
