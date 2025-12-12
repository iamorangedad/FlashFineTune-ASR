import logging
import sys
import json


class JsonFormatter(logging.Formatter):
    """
    将日志输出为 JSON 格式，方便 Loki/Grafana 解析
    """

    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "service": getattr(record, "service_name", "unknown"),
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            # 关键：尝试获取 extra 里的字段
            "req_id": getattr(record, "req_id", "N/A"),
            "session_id": getattr(record, "session_id", "N/A"),
        }
        return json.dumps(log_record, ensure_ascii=False)


def setup_logger(service_name):
    logger = logging.getLogger(service_name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 Handler
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # 使用 JSON 格式
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # 给 logger 绑定默认的 service_name
        logger = logging.LoggerAdapter(logger, {"service_name": service_name})

    return logger
