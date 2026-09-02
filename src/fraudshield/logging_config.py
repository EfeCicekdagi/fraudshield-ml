import logging
import sys
import os
from pythonjsonlogger import jsonlogger

SENSITIVE_FIELDS = {"amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "X-API-Key"}

class SensitiveDataFilter(logging.Filter):
    def filter(self, record):
        if hasattr(record, 'args') and isinstance(record.args, dict):
            # Redact sensitive keys in dict args
            record.args = {
                k: ("***REDACTED***" if k in SENSITIVE_FIELDS else v)
                for k, v in record.args.items()
            }
        # Redact in msg if it's a dict (common in json logging)
        if isinstance(record.msg, dict):
            record.msg = {
                k: ("***REDACTED***" if k in SENSITIVE_FIELDS else v)
                for k, v in record.msg.items()
            }
        return True

def setup_logger(name: str = "fraudshield") -> logging.Logger:
    """Setup and return a standard logger for the project."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        log_format = os.environ.get("LOG_FORMAT", "text").lower()
        console_handler = logging.StreamHandler(sys.stdout)
        
        if log_format == "json":
            formatter = jsonlogger.JsonFormatter(
                '%(asctime)s %(levelname)s %(name)s %(message)s'
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            
        console_handler.setFormatter(formatter)
        console_handler.addFilter(SensitiveDataFilter())
        logger.addHandler(console_handler)
        
        logger.addHandler(console_handler)
        
        # Prevent double logging if root logger is also configured
        logger.propagate = False
        
    return logger
