from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import zipfile

SRC = Path('/data/local-data/data/code/yunhengwang/paper/kimodo-policy')
DST = Path('/data/local-data/data/code/yunhengwang/paper/iclr2027_v1')
AUDIT = DST / 'migration_audit'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def text(root, name):
    return (root / name).read_text(encoding='utf-8')


def uncomment(s):
    return re.sub(r'(?<!\\)%[^\n]*', '', s)


def flatten(root, name):
    s = uncomment(text(root, name))
    return re.sub(r'\\(?:input|include)\{([^}]+)\}',
                  lambda m: flatten(root, m[1] if Path(m[1]).suffix else m[1] + '.tex'), s)


baseline = json.loads((AUDIT / 'source_manifest_before.json').read_text())
current = {}
for p in sorted(SRC.rglob('*')):
    if p.is_file():
        st = p.stat()
        current[p.relative_to(SRC).as_posix()] = {
            'size': st.st_size, 'sha256': digest(p), 'mtime_ns': st.st_mtime_ns}
assert baseline == current, 'Source contents or modification times changed'

copied = []
for folder in ['content', 'table', 'image', 'logo']:
    for p in sorted((SRC / folder).iterdir()):
        if p.is_file() and (folder in ['image', 'logo'] or p.suffix == '.tex'):
            name = p.relative_to(SRC).as_posix()
            if name in ['content/Abstract.tex', 'content/Experiments_Results.tex', 'content/Method.tex']:
                continue
            assert p.read_bytes() == (DST / name).read_bytes(), name
            copied.append(name)
assert (SRC / 'ref.bib').read_bytes() == (DST / 'ref.bib').read_bytes()

old_abstract = text(SRC, 'content/Abstract.tex')
old_teaser, old_abstract = old_abstract.split('\\begin{hclAbstract}', 1)
assert old_teaser.strip() == text(DST, 'content/Teaser.tex').strip()
old_abstract = old_abstract.replace('\\noindent\\textbf{Abstract.}\\quad\n', '', 1)
old_abstract = '\\begin{abstract}' + old_abstract.replace('\\end{hclAbstract}', '\\end{abstract}', 1)
assert old_abstract == text(DST, 'content/Abstract.tex')
old_results = text(SRC, 'content/Experiments_Results.tex')
override = r'''\titleformat{\subsection}
  {\normalfont\fontsize{14pt}{17pt}\selectfont\bfseries\color{black}}
  {\thesubsection}{0.55em}{}
'''
expected_results = old_results.replace(override, '% Use the official ICLR subsection formatting.\n')
opening, rest = expected_results.split('\\subsection{Evaluation on the HumanoidArena Benchmark}', 1)
expected_results = opening.rstrip() + ' Details of the datasets, training protocols, and evaluation environments are provided in Appendix~\\ref{sec:experimental_setup}.' + '\n\n\\subsection{Evaluation on the HumanoidArena Benchmark}' + rest
assert expected_results == text(DST, 'content/Experiments_Results.tex')

old_body = flatten(SRC, 'main_arxiv.tex').split('\\begin{document}', 1)[1]
new_body = flatten(DST, 'main.tex').split('\\begin{document}', 1)[1]
assert len(re.findall(r'\\appendix\b', new_body)) == 1
assert new_body.index('\\bibliography{ref}') < new_body.index('\\appendix') < new_body.index('\\section{Preliminaries}')
assert 'Project Page' not in new_body and 'Hugging Face' not in new_body and 'GitHub' not in new_body
old_method=text(SRC,'content/Method.tex')
first,rest=old_method.split('\n\n',1)
assert text(DST,'content/Method.tex') == first + ' Background on Kimodo and ControlNet is provided in Appendix~\\ref{sec:preliminaries}.' + '\n\n' + rest
assert new_body.index('\\section{Conclusion}') < new_body.index('\\bibliography{ref}')
assert new_body.index('\\appendix') < new_body.index('\\section{Preliminaries}') < new_body.index('\\section{Experimental Setup}')

patterns = {
    'figure_count': r'\\begin\{figure\*?\}',
    'table_count': r'\\begin\{table\*?\}',
    'equation_count': r'\\begin\{equation\*?\}',
    'labels': r'\\label\{([^}]+)\}',
    'citation_groups': r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',
    'reference_targets': r'\\(?:ref|eqref|autoref)\{([^}]+)\}',
    'section_titles': r'\\(?:sub)*section\*?\{([^}]+)\}',
}
result = {'source_unchanged_files': len(current), 'byte_identical_copied_files': len(copied) + 1}
for name, pattern in patterns.items():
    before = re.findall(pattern, old_body)
    after = re.findall(pattern, new_body)
    if name == 'reference_targets': before.extend(['sec:preliminaries', 'sec:experimental_setup'])
    assert Counter(before) == Counter(after), name
    result[name] = len(after)

old_graphics = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', old_body)
new_graphics = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', new_body)
assert Counter(old_graphics) <= Counter(new_graphics)
for name in new_graphics:
    assert (DST / name).is_file() and (DST / name).stat().st_size > 0, name
result['scientific_figures_preserved'] = len(old_graphics)
result['inactive_project_badges_preserved'] = len(new_graphics) - len(old_graphics)

official_files = ['iclr2027_conference.sty', 'iclr2027_conference.bst', 'natbib.sty', 'fancyhdr.sty', 'math_commands.tex']
with zipfile.ZipFile(DST / 'iclr-2027-style-files.zip') as z:
    for name in official_files:
        assert z.read('iclr2027/' + name) == (DST / name).read_bytes(), name
    for info in z.infolist():
        if not info.is_dir():
            assert z.read(info) == (DST / info.filename).read_bytes(), info.filename
result['official_template_files_unchanged'] = True

old_bibkeys = set(re.findall(r'\\bibcite\{([^}]+)\}', text(SRC, 'main_arxiv.aux')))
new_bibkeys = set(re.findall(r'\\bibcite\{([^}]+)\}', text(DST, 'main.aux')))
assert old_bibkeys == new_bibkeys, 'Compiled bibliography entries differ'
result['compiled_bibliography_entries_preserved'] = len(new_bibkeys)

log = text(DST, 'main.log')
errors = [line for line in log.splitlines() if
          line.startswith('!') or 'undefined' in line.lower() or
          'multiply defined' in line.lower() or 'Missing character:' in line]
result['compile_errors_or_undefined_references'] = errors
assert not errors, errors
result['overfull_box_warnings'] = [line for line in log.splitlines() if 'Overfull' in line]
result['pdf_sha256'] = digest(DST / 'main.pdf')
result['all_content_in_main_body'] = False
result['appendices'] = {'A': 'Preliminaries (Kimodo and ControlNet)', 'B': 'Experimental Setup'}
result['resource_badges_removed'] = True
(AUDIT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
(AUDIT / 'source_manifest_after.json').write_text(json.dumps(current, indent=2) + '\n')
print(json.dumps(result, indent=2))
