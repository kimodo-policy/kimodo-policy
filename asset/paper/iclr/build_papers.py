#!/usr/bin/env python3
"""Build independently cited main and supplementary PDFs and refresh cross-references."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / 'migration_audit'
XREFS = ROOT / 'crossrefs'

def update(path, text):
    if path.exists() and path.read_text() == text:
        return False
    path.write_text(text)
    return True

def labels(job):
    return [line for line in (ROOT / (job + '.aux')).read_text().splitlines()
            if line.startswith('\\newlabel{')]

def sync(job):
    lines = labels(job)
    changed = update(XREFS / (job + '_labels.aux'), '\n'.join(lines) + '\n')
    if job == 'main_text':
        text = '\n'.join(lines)
        counters = {}
        for counter, anchor in [('figure', 'figure'), ('table', 'table'), ('equation', 'equation')]:
            nums = [int(n) for n in re.findall(r'\{' + anchor + r'\.(\d+)\}', text)]
            counters[counter] = max(nums, default=0)
        content = '% Generated from main_text.aux; retain numbering across both documents.\n'
        content += ''.join('\\setcounter{%s}{%d}\n' % item for item in counters.items())
        changed |= update(XREFS / 'appendix_counters.tex', content)
    return changed

def build(job):
    log = AUDIT / ('independent_' + job + '_build.log')
    with log.open('w') as stream:
        subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', job + '.tex'],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print('Built ' + job, flush=True)

AUDIT.mkdir(exist_ok=True)
XREFS.mkdir(exist_ok=True)
for cycle in range(4):
    build('main_text')
    changed = sync('main_text')
    build('appendix')
    changed |= sync('appendix')
    if not changed:
        break
else:
    raise RuntimeError('Cross-reference metadata did not stabilize within four cycles.')

report = {'mode': 'independent LaTeX compilation with separate bibliographies', 'parts': {},
          'cross_reference_cycles': cycle + 1}
for job in ['main_text', 'appendix']:
    aux = (ROOT / (job + '.aux')).read_text()
    cited = {key.strip() for block in re.findall(r'\\citation\{([^}]+)\}', aux) for key in block.split(',')}
    listed = set(re.findall(r'\\bibcite\{([^}]+)\}', aux))
    assert cited == listed, (job, 'Citation/bibliography mismatch', cited ^ listed)
    log = (ROOT / (job + '.log')).read_text()
    failures = [line for line in log.splitlines() if re.search(
        r'undefined|multiply defined|Overfull|LaTeX Error|^!|destination with the same identifier|does not exist', line, re.I)]
    assert not failures, (job, failures)
    info = subprocess.check_output(['pdfinfo', str(ROOT / (job + '.pdf'))], text=True)
    pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M)[1])
    report['parts'][job] = {'pages': pages, 'cited_keys': sorted(cited), 'reference_count': len(listed),
                           'sha256': hashlib.sha256((ROOT / (job + '.pdf')).read_bytes()).hexdigest()}
for path in XREFS.glob('*_labels.aux'):
    assert all(line.startswith('\\newlabel{') for line in path.read_text().splitlines())
report['verification'] = 'Separate citations match each own bibliography; no undefined references, duplicate labels, missing destinations, or overfull boxes.'
(AUDIT / 'independent_documents_verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
