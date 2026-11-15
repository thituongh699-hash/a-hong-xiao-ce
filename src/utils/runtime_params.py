from src.utils.config_loader import ConfigLoader
from contextlib import contextmanager
from contextvars import ContextVar

_task_overrides = ContextVar('runtime_parameter_overrides', default=None)


@contextmanager
def parameter_overrides(values):
    """Apply public task settings without changing shared disk configuration."""
    token = _task_overrides.set(dict(values or {}))
    try:
        yield
    finally:
        _task_overrides.reset(token)

def get_value(path, default=None, reload_config=True):
    overrides = _task_overrides.get()
    if overrides is not None and path in overrides:
        return overrides[path]
    cfg = ConfigLoader.reload() if reload_config else ConfigLoader()
    return cfg.get(path, default)
