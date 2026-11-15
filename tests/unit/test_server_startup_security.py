import importlib
import os
from types import SimpleNamespace

import pytest

from src.utils import dependency_bootstrap as bootstrap


# Importing the server should never make this test invoke pip, even if the
# interpreter used to run the suite is missing an optional project package.
_done_env = bootstrap._BOOTSTRAP_DONE_ENV
_previous_done = os.environ.get(_done_env)
os.environ[_done_env] = "1"
try:
    server = importlib.import_module("server")
finally:
    if _previous_done is None:
        os.environ.pop(_done_env, None)
    else:
        os.environ[_done_env] = _previous_done


@pytest.fixture
def project_root(tmp_path):
    (tmp_path / "requirements.txt").write_text(
        "alpha-package==1.0\nbeta-package>=2\n", encoding="utf-8"
    )
    return tmp_path


@pytest.mark.parametrize("auto_install", [None, "0"])
def test_missing_dependencies_default_to_manual_install(
    project_root, monkeypatch, capsys, auto_install
):
    monkeypatch.delenv(bootstrap._BOOTSTRAP_DONE_ENV, raising=False)
    monkeypatch.setattr(bootstrap, "_is_locked_runtime", lambda: True)
    if auto_install is None:
        monkeypatch.delenv(bootstrap._BOOTSTRAP_ENABLED_ENV, raising=False)
    else:
        monkeypatch.setenv(bootstrap._BOOTSTRAP_ENABLED_ENV, auto_install)
    checked = []

    def missing(names):
        checked.append(names)
        return ["alpha-package", "beta-package"]

    monkeypatch.setattr(bootstrap, "_find_missing_requirements", missing)
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("default startup must not invoke pip"),
    )

    with pytest.raises(RuntimeError) as error:
        bootstrap.ensure_project_dependencies(project_root)

    output = capsys.readouterr().out + str(error.value)
    assert checked == [["alpha-package", "beta-package"]]
    assert "alpha-package, beta-package" in output
    assert "-m pip install -r" in output
    assert str(project_root / "requirements.txt") in output
    assert os.environ.get(bootstrap._BOOTSTRAP_DONE_ENV) != "1"


def test_explicit_auto_install_uses_existing_pip_bootstrap(project_root, monkeypatch):
    monkeypatch.delenv(bootstrap._BOOTSTRAP_DONE_ENV, raising=False)
    monkeypatch.setenv(bootstrap._BOOTSTRAP_ENABLED_ENV, "1")
    monkeypatch.setattr(bootstrap, "_is_locked_runtime", lambda: True)
    lock_path = project_root / "requirements.lock.txt"
    lock_path.write_text("alpha-package==1.0\nbeta-package==2.0\n", encoding="utf-8")
    missing_checks = iter([["alpha-package"], []])
    monkeypatch.setattr(
        bootstrap, "_find_missing_requirements", lambda names: next(missing_checks)
    )
    versions = {"alpha-package": "1.0", "beta-package": "2.0"}
    monkeypatch.setattr(
        bootstrap.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(version=versions[name]),
    )
    pip_calls = []
    monkeypatch.setattr(bootstrap, "_ensure_pip_available", lambda executable, root: None)
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda args, **kwargs: pip_calls.append((args, kwargs))
        or SimpleNamespace(returncode=0),
    )

    bootstrap.ensure_project_dependencies(project_root)

    assert len(pip_calls) == 1
    assert pip_calls[0][0][1:3] == ["-m", "pip"]
    assert pip_calls[0][0][3:5] == ["install", "-r"]
    assert pip_calls[0][0][-1] == str(lock_path)
    assert os.environ[bootstrap._BOOTSTRAP_DONE_ENV] == "1"


def test_preferred_lockfile_version_mismatch_is_reported_without_pip(
    project_root, monkeypatch, capsys
):
    monkeypatch.delenv(bootstrap._BOOTSTRAP_DONE_ENV, raising=False)
    monkeypatch.delenv(bootstrap._BOOTSTRAP_ENABLED_ENV, raising=False)
    monkeypatch.setattr(bootstrap, "_is_locked_runtime", lambda: True)
    lock_path = project_root / "requirements.lock.txt"
    lock_path.write_text("alpha-package==1.0\nbeta-package==2.0\n", encoding="utf-8")
    checked_names = []
    monkeypatch.setattr(
        bootstrap,
        "_find_missing_requirements",
        lambda names: checked_names.extend(names) or [],
    )
    versions = {"alpha-package": "1.1", "beta-package": "2.0"}
    monkeypatch.setattr(
        bootstrap.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(version=versions[name]),
    )
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("version mismatch must not invoke pip"),
    )

    with pytest.raises(RuntimeError, match=r"alpha-package==1\.0 \(当前 1\.1\)"):
        bootstrap.ensure_project_dependencies(project_root)

    output = capsys.readouterr().out
    assert checked_names == ["alpha-package", "beta-package"]
    assert str(lock_path) in output
    assert "-m pip install -r" in output


def test_lockfile_parser_accepts_exact_pins_with_extras(project_root):
    lock_path = project_root / "requirements.lock.txt"
    lock_path.write_text("uvicorn[standard]==0.51.0\n", encoding="utf-8")

    assert bootstrap._read_locked_requirements(lock_path) == {"uvicorn": "0.51.0"}


def test_other_runtime_uses_requirements_even_when_lock_exists(project_root, monkeypatch):
    monkeypatch.delenv(bootstrap._BOOTSTRAP_DONE_ENV, raising=False)
    monkeypatch.delenv(bootstrap._BOOTSTRAP_ENABLED_ENV, raising=False)
    monkeypatch.setattr(bootstrap, "_is_locked_runtime", lambda: False)
    (project_root / "requirements.lock.txt").write_text(
        "different-package==9.9\n", encoding="utf-8"
    )
    checked_names = []
    monkeypatch.setattr(
        bootstrap,
        "_find_missing_requirements",
        lambda names: checked_names.extend(names) or [],
    )
    monkeypatch.setattr(
        bootstrap.importlib.metadata,
        "distribution",
        lambda name: pytest.fail("the broad requirements list has no exact pins"),
    )
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("a clean fallback environment must not invoke pip"),
    )

    bootstrap.ensure_project_dependencies(project_root)

    assert checked_names == ["alpha-package", "beta-package"]
    assert os.environ[bootstrap._BOOTSTRAP_DONE_ENV] == "1"


def test_server_host_defaults_to_loopback_and_keeps_explicit_overrides(monkeypatch):
    monkeypatch.delenv("SERVER_HOST", raising=False)
    assert server._server_host({}) == "127.0.0.1"
    assert server._server_host({"system.server_host": "0.0.0.0"}) == "0.0.0.0"

    monkeypatch.setenv("SERVER_HOST", "192.0.2.10")
    assert server._server_host({"system.server_host": "127.0.0.1"}) == "192.0.2.10"
