#!/usr/bin/env python3
"""Build and verify a share ZIP from committed source and explicit evidence paths."""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'SCC_research_program_v0.1/'
BLOCKED = {'.git', '.venv', '__pycache__', '.pytest_cache', 'private', 'data',
           'runs', '.env', '.DS_Store', '.local-archives', '.storage-migrations'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def safe_name(name, supplemental=False):
    p = PurePosixPath(name)
    if (p.is_absolute() or '..' in p.parts or any(x in BLOCKED or x.startswith('._') for x in p.parts)
            or (not supplemental and 'artifacts' in p.parts)
            or name.endswith(('.pt', '.pth', '.safetensors', '.pem', '.key', '.zip', '.tar', '.tar.gz'))):
        raise ValueError(f'Excluded share member: {name}')


def verify(path, members):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != {PREFIX + n for n in members}:
            raise ValueError('Archive membership differs from the manifest')
        if z.testzip() is not None:
            raise ValueError('ZIP CRC check failed')
        for name, data in members.items():
            if digest(z.read(PREFIX + name)) != digest(data):
                raise ValueError(f'Archive hash mismatch: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backup-dir', type=Path, required=True)
    args = parser.parse_args()
    if git('status', '--porcelain').strip():
        raise SystemExit('Commit the working tree before building the share ZIP.')
    commit = git('rev-parse', 'HEAD').decode().strip()
    selection = json.loads(git('show', f'{commit}:configs/share.json'))
    source, modes = {}, {}
    with zipfile.ZipFile(BytesIO(git('archive', '--format=zip', commit))) as z:
        for entry in z.infolist():
            if entry.is_dir():
                continue
            safe_name(entry.filename)
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f'Source symlink is not shareable: {entry.filename}')
            source[entry.filename] = z.read(entry)
            modes[entry.filename] = mode
    supplement = {}
    total = 0
    for relative in selection['selected_evidence']:
        safe_name(relative, supplemental=True)
        if not relative.startswith('artifacts/'):
            raise ValueError(f'Supplement must be inside artifacts: {relative}')
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        files = [path] if path.is_file() else sorted(path.rglob('*'))
        for p in files:
            if any(part.startswith('._') or part in {'.DS_Store', '__pycache__', '.pytest_cache', 'pytest-tmp'}
                   for part in p.relative_to(ROOT).parts):
                continue
            if p.is_symlink():
                raise ValueError(f'Supplementary symlink: {p}')
            if not p.is_file():
                continue
            name = p.relative_to(ROOT).as_posix()
            safe_name(name, supplemental=True)
            if name in source or name in supplement:
                raise ValueError(f'Duplicate member: {name}')
            total += p.stat().st_size
            if total > selection['supplement_limit_bytes']:
                raise ValueError('Supplement exceeds the declared size limit')
            supplement[name] = p.read_bytes()
    for name, data in (source | supplement).items():
        signed_url = re.search(rb'https?://[^\s\x22\x27<>]+[?&]X-Amz-(?:Signature|Credential)=', data)
        key_block = re.search(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----[\r\n]', data)
        if signed_url or key_block:
            raise ValueError(f'Potential private material in {name}; review before sharing')
    metadata = {
        'source_commit': commit,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'start_here': 'README.md',
        'current_assessment': 'labnotes.md#current-position',
        'scope': 'Committed source and documentation plus explicitly selected small evidence.',
        'evidence_scope': selection['evidence_scope'],
        'evidence_limit': selection['evidence_limit'],
        'selected_evidence': selection['selected_evidence'],
        'source_sha256': {n: digest(b) for n, b in sorted(source.items())},
        'supplemental_sha256': {n: digest(b) for n, b in sorted(supplement.items())},
    }
    members = source | supplement
    members['_SHARE_INFO.json'] = (json.dumps(metadata, indent=2) + '\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.scc-share-', dir=args.output.parent) as tmp:
        staged = Path(tmp) / 'verified.zip'
        with zipfile.ZipFile(staged, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for name, data in sorted(members.items()):
                info = zipfile.ZipInfo(PREFIX + name, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = modes.get(name, 0o100644) << 16
                z.writestr(info, data)
        verify(staged, members)
        backup = None
        if args.output.exists():
            old_hash = digest(args.output.read_bytes())
            args.backup_dir.mkdir(parents=True, exist_ok=True)
            backup = args.backup_dir / f'{args.output.stem}-{old_hash}.zip'
            if not backup.exists():
                shutil.copyfile(args.output, backup)
            if digest(backup.read_bytes()) != old_hash:
                raise ValueError('Previous share ZIP backup failed verification')
        receipt = {'source_commit': commit, 'sha256': digest(staged.read_bytes()),
                   'bytes': staged.stat().st_size, 'source_files': len(source),
                   'supplement_files': len(supplement), 'members': len(members),
                   'all_hashes_verified': True, 'crc_verified': True,
                   'previous_zip_backup': str(backup) if backup else None}
        os.replace(staged, args.output)
        receipt_path = args.output.with_suffix('.receipt.json')
        staged_receipt = Path(tmp) / 'receipt.json'
        staged_receipt.write_text(json.dumps(receipt, indent=2) + '\n')
        os.replace(staged_receipt, receipt_path)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
