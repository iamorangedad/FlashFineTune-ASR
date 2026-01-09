import logging
import sys
import json


class JsonFormatter(logging.Formatter):
    """
    Format logs as JSON for Loki/Grafana
    """

    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "service": getattr(record, "service_name", "unknown"),
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            # These will now work correctly
            "req_id": getattr(record, "req_id", "N/A"),
            "session_id": getattr(record, "session_id", "N/A"),
        }
        return json.dumps(log_record, ensure_ascii=False)


# --- NEW: Custom Adapter to Merge Data ---
class MergeExtraAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        # Copy the default extra (service_name)
        new_extra = self.extra.copy() if self.extra else {}

        # Merge with the extra passed in the log call (session_id, req_id)
        if "extra" in kwargs:
            new_extra.update(kwargs["extra"])

        # Set the merged extra back into kwargs
        kwargs["extra"] = new_extra
        return msg, kwargs


def setup_logger(service_name):
    logger = logging.getLogger(service_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Use the Custom Adapter instead of the default one
    adapter = MergeExtraAdapter(logger, {"service_name": service_name})

    return adapter
