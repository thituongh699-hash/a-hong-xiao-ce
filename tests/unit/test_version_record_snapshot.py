from scripts import record_version


def test_snapshot_includes_private_config_regression_code_but_excludes_private_config(tmp_path, monkeypatch):
    files = {
        "tests/unit/test_config_loader_private_targets_regression.py": "# regression code\n",
        "src/config.private.json": '{"test_secret":"fixture-only"}',
        ".github/workflows/ci.yml": "name: test\n",
        "requirements.lock.txt": "example==1.0\n",
        ".python-version": "3.14.2\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(record_version, "ROOT", tmp_path)
    collected = {path.relative_to(tmp_path).as_posix() for path in record_version.source_files()}
    assert collected == set(files) - {"src/config.private.json"}
