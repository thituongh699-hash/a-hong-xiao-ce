import ast
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils.config_loader import ConfigLoader
from src.utils import dependency_bootstrap


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_targets_private_preferred_and_empty_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("CONFIG_PRIVATE_PATH", raising=False)

    config_path = tmp_path / "config.json"
    private_path = tmp_path / "config.private.json"

    _write_json(
        config_path,
        {
            "system": {"private_config_path": str(private_path)},
            "targets": ["600000.SH", "000001.SZ"],
        },
    )

    _write_json(private_path, {"targets": ["300750.SZ"]})
    cfg = ConfigLoader.reload(str(config_path))
    assert cfg.get("targets", []) == ["300750.SZ"]

    _write_json(private_path, {"targets": []})
    cfg = ConfigLoader.reload(str(config_path))
    assert cfg.get("targets", []) == ["600000.SH", "000001.SZ"]


def test_server_sse_dependency_bootstraps_before_import():
    requirement_names = dependency_bootstrap._read_requirement_names(PROJECT_ROOT / "requirements.txt")
    assert "sse-starlette" in requirement_names

    module = ast.parse((PROJECT_ROOT / "server.py").read_text(encoding="utf-8"))
    bootstrap_call_index = next(
        index
        for index, node in enumerate(module.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "ensure_project_dependencies"
    )
    sse_import_index = next(
        index
        for index, node in enumerate(module.body)
        if isinstance(node, ast.ImportFrom) and node.module == "sse_starlette.sse"
    )

    assert bootstrap_call_index < sse_import_index


@pytest.mark.parametrize("sse_installed,install_returncode", [(True, 0), (False, 0), (False, 1)])
def test_server_startup_handles_sse_dependency(monkeypatch, tmp_path, sse_installed, install_returncode):
    """Run the real startup prefix/bootstrap without pip, network or server side effects."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AHXC_DEPENDENCY_BOOTSTRAP_DONE", raising=False)
    monkeypatch.setenv("AHXC_AUTO_INSTALL_DEPS", "1")
    monkeypatch.setattr(dependency_bootstrap.sys, "frozen", False, raising=False)
    installed = sse_installed
    events = []
    selected_requirements, locked = dependency_bootstrap._select_requirements_file(PROJECT_ROOT)
    pins = dependency_bootstrap._read_locked_requirements(selected_requirements) if locked else {}

    def distribution(name):
        if name == "sse-starlette" and not installed:
            raise dependency_bootstrap.importlib.metadata.PackageNotFoundError(name)
        return SimpleNamespace(version=pins.get(name, "0"))

    def ensure_pip(executable, root):
        assert executable == dependency_bootstrap.sys.executable
        assert root == PROJECT_ROOT
        events.append("pip")

    def install(command, **kwargs):
        nonlocal installed
        assert command == [dependency_bootstrap.sys.executable, "-m", "pip", "install", "-r",
                           str(selected_requirements)]
        assert kwargs["cwd"] == str(PROJECT_ROOT)
        events.append("install")
        installed = install_returncode == 0
        return SimpleNamespace(returncode=install_returncode)

    original_import = builtins.__import__
    response_class = object()

    def guarded_import(name, *args, **kwargs):
        if name == "sse_starlette.sse":
            events.append("sse_import")
            if not installed:
                raise ModuleNotFoundError("No module named 'sse_starlette'")
            return SimpleNamespace(EventSourceResponse=response_class)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(dependency_bootstrap.importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_available", ensure_pip)
    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", install)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    source_path = PROJECT_ROOT / "server.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    sse_index = next(index for index, node in enumerate(module.body)
                     if isinstance(node, ast.ImportFrom) and node.module == "sse_starlette.sse")
    prefix = ast.Module(body=module.body[:sse_index + 1], type_ignores=[])
    namespace = {"__file__": str(source_path), "__name__": "startup_regression"}
    code = compile(prefix, str(source_path), "exec")
    if install_returncode:
        with pytest.raises(RuntimeError, match="依赖自动安装失败"):
            exec(code, namespace)
        assert events == ["pip", "install"]
        assert dependency_bootstrap.os.environ.get("AHXC_DEPENDENCY_BOOTSTRAP_DONE") != "1"
    else:
        exec(code, namespace)
        assert namespace["EventSourceResponse"] is response_class
        assert events == (["sse_import"] if sse_installed else ["pip", "install", "sse_import"])
