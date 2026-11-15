"""Save a local source snapshot, incremental diff and auditable upgrade record."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ('server.py', 'main.py', 'run_backtest.py', 'run_live.py', 'dashboard.html',
              '.gitignore', 'requirements.txt', 'requirements.lock.txt', '.python-version', 'README.md', 'desktop.spec', 'config.json',
              'docs/参数优化模块.md', 'docs/交易范围筛选.md', 'docs/upgrade-roadmap.md')
SUFFIXES = {'.py', '.html', '.js', '.cjs', '.json', '.toml', '.yml', '.yaml'}
IGNORED_DIRS = {'node_modules', '__pycache__', 'test-results', 'playwright-report'}


def source_files():
    files = []
    for name in ('src', 'scripts', 'tests', 'static', '.github'):
        for path in (ROOT / name).rglob('*'):
            if path.is_file() and path.suffix in SUFFIXES and not IGNORED_DIRS.intersection(path.parts):
                if not path.name.lower().startswith('config.private'):
                    files.append(path)
    files.extend(ROOT / name for name in ROOT_FILES if (ROOT / name).is_file())
    return sorted(set(files))


def record(version, description, notes_file, verification_file, previous):
    if not re.fullmatch(r'\d{4}\.\d{2}\.\d{2}\.\d+', version):
        raise ValueError('version must be YYYY.MM.DD.N')
    previous = previous.resolve()
    old = json.loads((previous / 'manifest.json').read_text(encoding='utf-8'))
    validation = json.loads(verification_file.read_text(encoding='utf-8'))
    if validation.get('status') != 'passed':
        raise ValueError('verification must have status=passed before saving a completed version')
    notes = notes_file.read_text(encoding='utf-8')
    folder = ROOT / 'reports' / 'upgrades' / version
    if folder.exists():
        raise ValueError('version exists; records are append-only')
    paths = source_files()
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    changed = [name for name in sorted(set(hashes) | set(old['files'])) if hashes.get(name) != old['files'].get(name)]
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / 'source-snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    patch = []
    with zipfile.ZipFile(previous / 'source-snapshot.zip') as before:
        old_names = set(before.namelist())
        for name in changed:
            # Local snapshots preserve config, but exported differences never
            # include configuration values or private files.
            if name == 'config.json' or Path(name).name.lower().startswith('config.private'):
                continue
            prior = before.read(name).decode('utf-8-sig').splitlines(keepends=True) if name in old_names else []
            current_path = ROOT / name
            current = current_path.read_text(encoding='utf-8-sig').splitlines(keepends=True) if current_path.exists() else []
            patch.extend(difflib.unified_diff(prior, current, fromfile='before/' + name, tofile='after/' + name))
    (folder / 'changes.patch').write_text(''.join(patch), encoding='utf-8')
    manifest = dict(version=version, previous_version=old['version'], description=description,
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    changed_files=changed, files=hashes, verification=validation,
                    snapshot_scope='source and tests; no private configuration, data, build or dependencies')
    (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (folder / 'verification.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    public = ROOT / 'docs' / 'versions'
    public.mkdir(parents=True, exist_ok=True)
    record_path = public / (version + '.md')
    if record_path.exists():
        raise ValueError('public record exists; refusing overwrite')
    details = '\n'.join('- `' + name + '`' for name in changed)
    record_path.write_text(f'# {version} · {description}\n\n{notes}\n\n'
                          f'验证记录：`reports/upgrades/{version}/verification.json`。\n'
                          f'源代码快照、差异和哈希：`reports/upgrades/{version}/`（仅本机保存）。\n\n'
                          f'变更文件：\n\n{details}\n', encoding='utf-8')
    index = public / 'index.md'
    if not index.exists():
        index.write_text('# 升级版本记录\n\n版本按 YYYY.MM.DD.N 顺序追加；每批通过验证后保存。'
                         '原有工作区快照见 reports/upgrades/2026-10-08/baseline。\n\n', encoding='utf-8')
    with index.open('a', encoding='utf-8') as stream:
        stream.write(f'- [{version} · {description}]({version}.md)\n')
    print(json.dumps({'version': version, 'changed_files': len(changed), 'record': str(record_path)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--description', required=True)
    parser.add_argument('--notes-file', type=Path, required=True)
    parser.add_argument('--verification-file', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    args = parser.parse_args()
    record(args.version, args.description, args.notes_file, args.verification_file, args.previous)
