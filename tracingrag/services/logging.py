"""Structured logging service"""

import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from uuid import uuid4

import structlog

# Context variable for request ID
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


def add_request_id(logger, method_name, event_dict):
    """Add request ID to log context"""
    request_id = request_id_ctx.get()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def add_timestamp(logger, method_name, event_dict):
    """Add ISO timestamp to log"""
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict


def add_log_level(logger, method_name, event_dict):
    """Add log level to event"""
    event_dict["level"] = method_name.upper()
    return event_dict


# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        add_timestamp,
        add_log_level,
        add_request_id,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    cache_logger_on_first_use=False,
)


class LoggerService:
    """Structured logging service with context support"""

    def __init__(self, component: str):
        """Initialize logger

        Args:
            component: Component name for logging context
        """
        self.component = component
        self.logger = structlog.get_logger(component=component)

    def _log(self, level: str, event: str, **kwargs):
        """Internal log method

        Args:
            level: Log level
            event: Event message
            **kwargs: Additional context
        """
        log_func = getattr(self.logger, level)
        log_func(event, **kwargs)

    def debug(self, event: str, **kwargs):
        """Log debug message"""
        self._log("debug", event, **kwargs)

    def info(self, event: str, **kwargs):
        """Log info message"""
        self._log("info", event, **kwargs)

    def warning(self, event: str, **kwargs):
        """Log warning message"""
        self._log("warning", event, **kwargs)

    def error(self, event: str, **kwargs):
        """Log error message"""
        self._log("error", event, **kwargs)

    def critical(self, event: str, **kwargs):
        """Log critical message"""
        self._log("critical", event, **kwargs)

    def bind(self, **kwargs) -> "LoggerService":
        """Bind context to logger

        Args:
            **kwargs: Context to bind

        Returns:
            New logger with bound context
        """
        new_logger = LoggerService(self.component)
        new_logger.logger = self.logger.bind(**kwargs)
        return new_logger


# Component-specific loggers
def get_logger(component: str) -> LoggerService:
    """Get logger for a component

    Args:
        component: Component name

    Returns:
        Logger service instance
    """
    return LoggerService(component)


# Pre-configured loggers for common components
api_logger = get_logger("api")
memory_logger = get_logger("memory")
rag_logger = get_logger("rag")
agent_logger = get_logger("agent")
promotion_logger = get_logger("promotion")
cache_logger = get_logger("cache")
consolidation_logger = get_logger("consolidation")
database_logger = get_logger("database")
llm_logger = get_logger("llm")


# Utility functions
def set_request_id(request_id: str | None = None):
    """Set request ID in context

    Args:
        request_id: Request ID (generates new if None)
    """
    if request_id is None:
        request_id = str(uuid4())
    request_id_ctx.set(request_id)
    return request_id


def get_request_id() -> str:
    """Get current request ID

    Returns:
        Request ID from context
    """
    return request_id_ctx.get()


def clear_request_id():
    """Clear request ID from context"""
    request_id_ctx.set("")


# Audit logging
class AuditLogger:
    """Audit logger for security-sensitive operations"""

    def __init__(self):
        self.logger = get_logger("audit")

    def log_auth_attempt(self, user: str, success: bool, ip: str, **kwargs):
        """Log authentication attempt"""
        self.logger.info(
            "auth_attempt",
            user=user,
            success=success,
            ip=ip,
            **kwargs,
        )

    def log_api_access(
        self, user: str, endpoint: str, method: str, status: int, **kwargs
    ):
        """Log API access"""
        self.logger.info(
            "api_access",
            user=user,
            endpoint=endpoint,
            method=method,
            status=status,
            **kwargs,
        )

    def log_memory_created(self, user: str, topic: str, state_id: str, **kwargs):
        """Log memory state creation"""
        self.logger.info(
            "memory_created",
            user=user,
            topic=topic,
            state_id=state_id,
            **kwargs,
        )

    def log_memory_updated(self, user: str, topic: str, state_id: str, **kwargs):
        """Log memory state update"""
        self.logger.info(
            "memory_updated",
            user=user,
            topic=topic,
            state_id=state_id,
            **kwargs,
        )

    def log_promotion(
        self, user: str, topic: str, trigger: str, success: bool, **kwargs
    ):
        """Log memory promotion"""
        self.logger.info(
            "promotion",
            user=user,
            topic=topic,
            trigger=trigger,
            success=success,
            **kwargs,
        )

    def log_query(self, user: str, query: str, use_agent: bool, **kwargs):
        """Log query execution"""
        self.logger.info(
            "query",
            user=user,
            query=query,
            use_agent=use_agent,
            **kwargs,
        )

    def log_security_event(self, event_type: str, severity: str, **kwargs):
        """Log security event"""
        self.logger.warning(
            "security_event",
            event_type=event_type,
            severity=severity,
            **kwargs,
        )


audit_logger = AuditLogger()


# Performance logging
class PerformanceLogger:
    """Performance logger for operation timing"""

    def __init__(self):
        self.logger = get_logger("performance")

    def log_operation(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        **kwargs,
    ):
        """Log operation performance

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            success: Whether operation succeeded
            **kwargs: Additional context
        """
        self.logger.info(
            "operation_performance",
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            **kwargs,
        )

    def log_query_performance(
        self,
        query_type: str,
        duration_ms: float,
        states_retrieved: int,
        **kwargs,
    ):
        """Log query performance"""
        self.logger.info(
            "query_performance",
            query_type=query_type,
            duration_ms=duration_ms,
            states_retrieved=states_retrieved,
            **kwargs,
        )

    def log_llm_performance(
        self,
        model: str,
        operation: str,
        duration_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        **kwargs,
    ):
        """Log LLM performance"""
        self.logger.info(
            "llm_performance",
            model=model,
            operation=operation,
            duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            **kwargs,
        )

    def log_cache_performance(
        self,
        cache_type: str,
        operation: str,
        hit: bool,
        duration_ms: float,
        **kwargs,
    ):
        """Log cache performance"""
        self.logger.info(
            "cache_performance",
            cache_type=cache_type,
            operation=operation,
            hit=hit,
            duration_ms=duration_ms,
            **kwargs,
        )


performance_logger = PerformanceLogger()
