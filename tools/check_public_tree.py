"""Redacted, offline publication audit; heuristic checks, not a security warranty.

History mode reads every reachable blob once through git cat-file. It never
prints matching text, credentials, commit messages or author identities.
This complements (does not replace) independent secret scanners and rights review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

RULES = {
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'openai_key': rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}',
    'github_token': rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
    'aws_access_key': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'credential_literal': rb'''(?i)(?:api[_-]?key|api[_-]?secret|password|access[_-]?token)\s*[:=]\s*["'][^"'\r\n]{16,}["']''',
    'personal_home_path': rb'(?:/Users/[A-Za-z][A-Za-z0-9_.-]+/|[A-Z]:\\{1,2}Users\\{1,2}[A-Za-z][A-Za-z0-9_.-]+\\)',
    'broker_account_identifier': rb'\b(?:DEMO|FFF)[0-9]{5,}\b',
}
PATTERNS = {name: re.compile(pattern) for name,pattern in RULES.items()}
PRIVATE_SUFFIXES = {'.dbn','.parquet','.sqlite3','.db','.pkl','.pickle','.log','.zip','.gz','.zst'}


def content_findings(data: bytes, identity: str) -> list[dict]:
    out = []
    for name,pattern in PATTERNS.items():
        for match in pattern.finditer(data):
            out.append({'rule':name,'object':identity,
                        'line':data[:match.start()].count(b'\n')+1,
                        'fingerprint':hashlib.sha256(match.group()).hexdigest()[:16]})
    return out


def path_findings(path: str) -> list[dict]:
    p = Path(path)
    out = []
    if p.name == '.env' or p.name.startswith('.env.') or p.suffix.lower() in {'.pem','.key','.p12','.pfx'}:
        out.append({'rule':'credential_file_path','path':path})
    if p.suffix.lower() in PRIVATE_SUFFIXES:
        out.append({'rule':'data_or_generated_artifact','path':path})
    if path.startswith(('.diag/','data/','runs/','tradingview/','ninjatrader/')):
        out.append({'rule':'private_research_or_unreviewed_data','path':path})
    return out


def git(repo: Path,*args: str) -> bytes:
    return subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True).stdout


def audit_history(repo: Path) -> dict:
    listing = git(repo,'rev-list','--objects','--all').decode().splitlines()
    objects = {line.split(' ',1)[0] for line in listing}
    paths = {line.split(' ',1)[1] for line in listing if ' ' in line}
    findings = [f for path in sorted(paths) for f in path_findings(path)]
    sizes = []
    process = subprocess.Popen(['git','-C',str(repo),'cat-file','--batch'],
                               stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    blobs = 0
    total = 0
    try:
        for oid in sorted(objects):
            process.stdin.write((oid+'\n').encode());process.stdin.flush()
            header = process.stdout.readline().decode().split()
            if len(header) != 3: raise ValueError('Invalid git object header')
            size = int(header[2]);data = process.stdout.read(size)
            if len(data) != size or process.stdout.read(1) != b'\n':
                raise ValueError('Incomplete Git object scan')
            if header[1] != 'blob': continue
            blobs += 1;total += size
            findings.extend(content_findings(data,oid))
            if size > 1_000_000: sizes.append({'object':oid,'bytes':size})
    finally:
        process.stdin.close();process.stdout.close();process.wait(timeout=10)
    if process.returncode: raise ValueError('git cat-file failed')
    return {'scope':'all_fetched_refs','refs':git(repo,'for-each-ref','--format=%(refname)').decode().splitlines(),
            'blobs_scanned':blobs,'bytes_scanned':total,'large_blobs':sizes,
            'findings':findings,'limits':'Heuristic patterns only; no external secret scanner or legal rights verification.'}


def audit_tree(root: Path) -> dict:
    findings = [];count = 0;total = 0
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            findings.append({'rule':'symlink_not_allowed','path':str(path.relative_to(root))});continue
        if not path.is_file(): continue
        rel = path.relative_to(root).as_posix()
        if any(part in {'.git','__pycache__','.venv','build'} or part.endswith('.egg-info') for part in path.relative_to(root).parts):
            continue
        data = path.read_bytes();count += 1;total += len(data)
        findings.extend(path_findings(rel));findings.extend(content_findings(data,rel))
    return {'scope':'public_tree','files_scanned':count,'bytes_scanned':total,
            'findings':findings,'limits':'Heuristic patterns only; no external secret scanner or legal rights verification.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--history',type=Path);group.add_argument('--tree',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    report = audit_history(args.history) if args.history else audit_tree(args.tree)
    text = json.dumps(report,indent=2,sort_keys=True)+'\n'
    if args.output:
        with args.output.open('x',encoding='utf-8') as f: f.write(text)
        print(json.dumps({'report':str(args.output),'findings':len(report['findings'])}))
    else: print(text,end='')
    return 1 if report['findings'] else 0


if __name__ == '__main__': raise SystemExit(main())
