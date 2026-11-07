"""Per-operation metrics scoped to the current execution context."""

import atexit
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter


_metrics: ContextVar[dict | None] = ContextVar("ingestion_metrics", default=None)
_observation: ContextVar[object | None] = ContextVar("langfuse_observation", default=None)
_shutdown_registered = False


def get_langfuse_client():
    """Return the optional Langfuse client; disabled tracing never imports its SDK."""
    from .settings import get_settings

    if not get_settings().langfuse_enabled:
        return None
    from langfuse import get_client

    client = get_client()
    global _shutdown_registered
    if not _shutdown_registered:
        atexit.register(flush_langfuse)
        _shutdown_registered = True
    return client


def create_trace_id() -> str | None:
    try:
        client = get_langfuse_client()
        return client.create_trace_id() if client is not None else None
    except Exception:
        return None


@contextmanager
def observation(
    name: str,
    *,
    as_type: str = "span",
    metadata: dict | None = None,
    trace_id: str | None = None,
    model: str | None = None,
    model_parameters: dict | None = None,
):
    """Create a content-free Langfuse span; tracing failures never break application work."""
    try:
        client = get_langfuse_client()
        if client is None:
            context = None
        else:
            options = {"name": name, "as_type": as_type, "metadata": metadata or {}}
            if trace_id:
                options["trace_context"] = {"trace_id": trace_id}
            if model:
                options["model"] = model
            if model_parameters:
                options["model_parameters"] = model_parameters
            context = client.start_as_current_observation(**options)
    except Exception:
        context = None

    if context is None:
        yield None
        return

    try:
        span = context.__enter__()
    except Exception:
        yield None
        return

    error_info = None
    token = _observation.set(span)
    try:
        yield span
    except BaseException as exc:
        error_info = sys.exc_info()
        try:
            span.update(level="ERROR", status_message=type(exc).__name__)
        except Exception:
            pass
    finally:
        _observation.reset(token)
        try:
            context.__exit__(*(error_info or (None, None, None)))
        except Exception:
            pass
    if error_info is not None:
        raise error_info[1].with_traceback(error_info[2])


def update_current_observation(**attributes) -> None:
    span = _observation.get()
    if span is not None:
        try:
            span.update(**attributes)
        except Exception:
            pass


def flush_langfuse() -> None:
    try:
        client = get_langfuse_client()
        if client is not None:
            client.flush()
    except Exception:
        pass


@contextmanager
def capture_metrics():
    metrics = {"stage_seconds": {}}
    token = _metrics.set(metrics)
    started = perf_counter()
    try:
        yield metrics
    finally:
        metrics["total_seconds"] = round(perf_counter() - started, 6)
        _metrics.reset(token)


@contextmanager
def stage(name: str):
    started = perf_counter()
    error = None
    with observation(name, metadata={"stage": name}) as span:
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            duration = perf_counter() - started
            metrics = _metrics.get()
            if metrics is not None:
                stages = metrics["stage_seconds"]
                stages[name] = round(stages.get(name, 0) + duration, 6)
            if span is not None:
                try:
                    span.update(output={"duration_seconds": round(duration, 6), "status": error or "ok"})
                except Exception:
                    pass


def measured(name, function, *args, **kwargs):
    with stage(name):
        return function(*args, **kwargs)


def increment_metric(name: str, amount: int = 1):
    metrics = _metrics.get()
    if metrics is not None:
        metrics[name] = metrics.get(name, 0) + amount


def record_usage(*, input_tokens: int | None = None, output_tokens: int | None = None,
                 total_tokens: int | None = None) -> None:
    usage = {}
    if input_tokens is not None:
        usage["input"] = input_tokens
    if output_tokens is not None:
        usage["output"] = output_tokens
    if total_tokens is not None:
        usage["total"] = total_tokens
    if usage:
        update_current_observation(usage_details=usage)