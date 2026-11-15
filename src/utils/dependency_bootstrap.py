"""项目启动依赖自检与自动安装工具。"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import List

# 使用环境变量避免同一进程重复执行依赖检查。
_BOOTSTRAP_DONE_ENV = "AHXC_DEPENDENCY_BOOTSTRAP_DONE"
# 仅在显式选择时安装依赖；默认启动只检查，不修改当前 Python 环境。
_BOOTSTRAP_ENABLED_ENV = "AHXC_AUTO_INSTALL_DEPS"
# 简单提取 requirements 包名，兼容常见版本约束写法。
_REQUIREMENT_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")
_PINNED_REQUIREMENT_PATTERN = re.compile(
    r"^\s*([A-Za-z0-9_.\-]+)(?:\[[A-Za-z0-9_,.\-]+\])?\s*==\s*([^\s;]+)"
)


def _normalize_distribution_name(name: str) -> str:
    """统一包名格式，减少大小写和分隔符差异带来的误判。"""
    return str(name or "").strip().lower().replace("_", "-")


def _project_root_from_current_file() -> Path:
    """基于当前文件位置推导项目根目录。"""
    return Path(__file__).resolve().parents[2]


def _is_locked_runtime() -> bool:
    """Return whether this interpreter matches the verified lockfile platform."""
    machine = platform.machine().strip().lower()
    return (
        sys.platform == "win32"
        and getattr(sys.implementation, "name", "") == "cpython"
        and tuple(sys.version_info[:3]) == (3, 14, 2)
        and machine in {"amd64", "x86_64"}
    )


def _select_requirements_file(project_root: Path) -> tuple[Path, bool]:
    """Select the verified Windows lock when available, otherwise the broad list."""
    lock_path = project_root / "requirements.lock.txt"
    if _is_locked_runtime() and lock_path.is_file():
        return lock_path, True
    return project_root / "requirements.txt", False


def _read_requirement_names(requirements_path: Path) -> List[str]:
    """从 requirements.txt 中提取包名列表。"""
    requirement_names: List[str] = []
    seen_names = set()

    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        # 跳过空行、注释和 include/editable 等扩展语法。
        if not line or line.startswith("#") or line.startswith("-"):
            continue

        # 去掉环境标记，例如 "package; python_version >= '3.9'"。
        line = line.split(";", 1)[0].strip()
        if not line:
            continue

        match = _REQUIREMENT_NAME_PATTERN.match(line)
        if not match:
            continue

        normalized_name = _normalize_distribution_name(match.group(1))
        if normalized_name and normalized_name not in seen_names:
            seen_names.add(normalized_name)
            requirement_names.append(normalized_name)

    return requirement_names


def _read_locked_requirements(requirements_path: Path) -> dict[str, str]:
    """Read exact name==version pins from the runtime lockfile."""
    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        requirements_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PINNED_REQUIREMENT_PATTERN.fullmatch(line)
        if not match:
            raise RuntimeError(
                f"锁文件格式错误: {requirements_path.name}:{line_number} "
                "必须使用 package==version 精确固定依赖。"
            )
        name = _normalize_distribution_name(match.group(1))
        version = match.group(2)
        previous = pins.get(name)
        if previous is not None and previous != version:
            raise RuntimeError(f"锁文件重复固定了不同版本: {name}=={previous} / {version}")
        pins[name] = version
    return pins


def _find_missing_requirements(requirement_names: List[str]) -> List[str]:
    """找出当前解释器环境中缺失的依赖包。"""
    missing_names: List[str] = []

    for requirement_name in requirement_names:
        try:
            importlib.metadata.distribution(requirement_name)
        except importlib.metadata.PackageNotFoundError:
            missing_names.append(requirement_name)

    return missing_names


def _find_version_mismatches(pins: dict[str, str]) -> List[str]:
    """Find installed distributions that do not match exact lockfile versions."""
    mismatches: List[str] = []
    for name, expected in pins.items():
        try:
            actual = importlib.metadata.distribution(name).version
        except importlib.metadata.PackageNotFoundError:
            continue
        if str(actual) != expected:
            mismatches.append(f"{name}=={expected} (当前 {actual})")
    return mismatches


def _ensure_pip_available(python_executable: str, project_root: Path) -> None:
    """确保当前解释器具备 pip，缺失时尝试自动补齐。"""
    pip_check = subprocess.run(
        [python_executable, "-m", "pip", "--version"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if pip_check.returncode == 0:
        return

    print("[bootstrap] pip 不可用，正在尝试通过 ensurepip 自动安装...")
    ensurepip_result = subprocess.run(
        [python_executable, "-m", "ensurepip", "--upgrade"],
        cwd=str(project_root),
        check=False,
    )
    if ensurepip_result.returncode != 0:
        raise RuntimeError("当前 Python 环境缺少 pip，且 ensurepip 自动修复失败。")


def ensure_project_dependencies(project_root: str | Path | None = None) -> None:
    """Check project dependencies; install them only when explicitly enabled."""
    if os.environ.get(_BOOTSTRAP_DONE_ENV) == "1":
        return

    # 冻结打包环境的依赖通常已随程序分发，不在运行时执行 pip。
    if getattr(sys, "frozen", False):
        os.environ[_BOOTSTRAP_DONE_ENV] = "1"
        return

    resolved_project_root = Path(project_root) if project_root else _project_root_from_current_file()
    requirements_path, is_locked = _select_requirements_file(resolved_project_root)
    if not requirements_path.exists():
        os.environ[_BOOTSTRAP_DONE_ENV] = "1"
        return

    if is_locked:
        pins = _read_locked_requirements(requirements_path)
        requirement_names = list(pins)
    else:
        pins = {}
        requirement_names = _read_requirement_names(requirements_path)
    if not requirement_names:
        os.environ[_BOOTSTRAP_DONE_ENV] = "1"
        return

    missing_names = _find_missing_requirements(requirement_names)
    version_mismatches = _find_version_mismatches(pins) if is_locked else []
    if not missing_names and not version_mismatches:
        os.environ[_BOOTSTRAP_DONE_ENV] = "1"
        return

    python_executable = sys.executable or "python"
    problems = []
    if missing_names:
        problems.append(f"缺少: {', '.join(missing_names)}")
    if version_mismatches:
        problems.append(f"版本不符: {', '.join(version_mismatches)}")
    print(f"[bootstrap] 依赖检查失败: {'; '.join(problems)}")
    if str(os.environ.get(_BOOTSTRAP_ENABLED_ENV, "")).strip() != "1":
        command = f'"{python_executable}" -m pip install -r "{requirements_path}"'
        message = (
            f"项目依赖与所选清单不匹配: {'; '.join(problems)}。默认启动不会自动安装依赖。\n"
            f"请手动执行: {command}\n"
            f"或设置 {_BOOTSTRAP_ENABLED_ENV}=1 后重试以允许自动安装。"
        )
        print(f"[bootstrap] {message}")
        raise RuntimeError(message)

    print(f"[bootstrap] 正在使用 {python_executable} 自动安装 {requirements_path.name} ...")

    _ensure_pip_available(python_executable, resolved_project_root)

    install_result = subprocess.run(
        [python_executable, "-m", "pip", "install", "-r", str(requirements_path)],
        cwd=str(resolved_project_root),
        check=False,
    )
    if install_result.returncode != 0:
        command = f'"{python_executable}" -m pip install -r "{requirements_path}"'
        raise RuntimeError(
            f"依赖自动安装失败，请手动执行 `{command}` 后重试。"
        )

    remaining_missing_names = _find_missing_requirements(requirement_names)
    remaining_version_mismatches = _find_version_mismatches(pins) if is_locked else []
    if remaining_missing_names or remaining_version_mismatches:
        remaining_problems = []
        if remaining_missing_names:
            remaining_problems.append(f"缺少: {', '.join(remaining_missing_names)}")
        if remaining_version_mismatches:
            remaining_problems.append(f"版本不符: {', '.join(remaining_version_mismatches)}")
        raise RuntimeError(
            "依赖安装后检查仍未通过: {}".format("; ".join(remaining_problems))
        )

    print("[bootstrap] 依赖检查完成，所有 requirements 依赖均已就绪。")
    os.environ[_BOOTSTRAP_DONE_ENV] = "1"
