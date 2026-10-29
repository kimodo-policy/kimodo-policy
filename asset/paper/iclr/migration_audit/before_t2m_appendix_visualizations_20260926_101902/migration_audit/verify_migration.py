from pathlib import Path
from collections import Counter
import hashlib,json,re,zipfile,subprocess
DST=Path('/data/local-data/data/code/yunhengwang/paper/iclr2027_v1')
SRC=Path('/data/local-data/data/code/yunhengwang/paper/kimodo-policy')
AUDIT=DST/'migration_audit'
checkpoint=Path((AUDIT/'latest_revision_checkpoint.txt').read_text().strip())
assert checkpoint.is_relative_to(AUDIT)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
original=json.loads((AUDIT/'source_manifest_before.json').read_text())
current={}
for p in SRC.rglob('*'):
 if p.is_file():
  st=p.stat(); current[p.relative_to(SRC).as_posix()]={'size':st.st_size,'sha256':digest(p),'mtime_ns':st.st_mtime_ns}
assert current==original,'Original arXiv directory changed'
before=json.loads((checkpoint/'target_manifest_before.json').read_text())
expected=json.loads((checkpoint/'expected_edits.json').read_text())
for name,sha in before.items():
 assert digest(DST/name)==expected.get(name,sha),name
assert set(expected)=={'content/Introduction.tex'}
assert '\\input{content/Teaser}\n\\input{content/Abstract}\n\\clearpage' in (DST/'main.tex').read_text()
def flatten(name,old=False):
 p=checkpoint/name if old and (checkpoint/name).is_file() else DST/name
 s=re.sub(r'(?<!\\)%[^\n]*','',p.read_text())
 return re.sub(r'\\(?:input|include)\{([^}]+)\}',lambda m:flatten(m[1] if Path(m[1]).suffix else m[1]+'.tex',old),s)
body=flatten('main.tex').split('\\begin{document}',1)[1]
oldbody=flatten('main.tex',True).split('\\begin{document}',1)[1]
report={'source_unchanged_files':len(current),'edited_tex_files':list(expected),'other_current_project_files_unchanged':True}




oldintro=(checkpoint/'content/Introduction.tex').read_text()
newintro=(DST/'content/Introduction.tex').read_text()
old=r'\suppressfloats[t]'+'\n'+r'\begin{figure*}[t]'
new='% Place the architecture comparison at the bottom of page 2.\n'+r'\begin{figure*}[!b]'
assert oldintro.replace(old,new)==newintro
report['changes_limited_to_figure_2_float_placement']=True
report['manuscript_wording_and_image_sizes_unchanged']=True
for kind in ['figure','table','equation']:
 pattern=r'\\begin\{'+kind+r'\*?\}'
 assert len(re.findall(pattern,body))==len(re.findall(pattern,oldbody))
 report[kind+'_count']=len(re.findall(pattern,body))
for name,pattern in [('labels',r'\\label\{([^}]+)\}'),('reference_targets',r'\\(?:ref|eqref|autoref)\{([^}]+)\}')]:
 assert Counter(re.findall(pattern,body))==Counter(re.findall(pattern,oldbody))
 report[name]=len(re.findall(pattern,body))
for image in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',body):
 assert (DST/image).is_file() and (DST/image).stat().st_size>0,image
assert 'image/teaser.png' not in body
assert 'Top: benchmark performance' not in body
assert 'red, blue, and green indicating ours' not in body
assert 'Center: diverse VLA pretraining' not in body
assert 'With 678M total and only 92M trainable parameters, Kimodo-Policy' in newintro
report['figure_1_asset']='image/teasser.png'
report['user_figure_1_image_unchanged']=digest(DST/'image/teasser.png')==before['image/teasser.png']
with zipfile.ZipFile(DST/'iclr-2027-style-files.zip') as z:
 for name in ['iclr2027_conference.sty','iclr2027_conference.bst','natbib.sty','fancyhdr.sty','math_commands.tex']:
  assert (DST/name).read_bytes()==z.read('iclr2027/'+name)
report['official_template_unchanged']=True
log=(DST/'main.log').read_text()
errors=[s for s in log.splitlines() if s.startswith('!') or 'undefined' in s.lower() or 'multiply defined' in s.lower() or 'Missing character:' in s]
assert not errors,errors
report['compile_errors_or_undefined_references']=errors
report['overfull_box_warnings']=[s for s in log.splitlines() if 'Overfull' in s]
aux=(DST/'main.aux').read_text()
assert re.search(r'\\newlabel\{fig:overview\}\{\{1\}\{1\}',aux)
report['figure_1_on_first_page']=True
report['compiled_bibliography_entries']=len(set(re.findall(r'\\bibcite\{([^}]+)\}',aux)))
info=subprocess.check_output(['pdfinfo',str(DST/'main.pdf')],text=True)
report['pdf_pages']=int(re.search(r'Pages:\s+(\d+)',info)[1])
report['pdf_sha256']=digest(DST/'main.pdf')
import xml.etree.ElementTree as ET
bbox=AUDIT/'main_bbox.html'
subprocess.run(['pdftotext','-bbox-layout',str(DST/'main.pdf'),str(bbox)],check=True)
ns={'x':'http://www.w3.org/1999/xhtml'}
pages=ET.parse(bbox).getroot().findall('.//x:page',ns)
rows=[]
for line in pages[0].findall('.//x:line',ns):
 if float(line.get('xMin'))>100:
  rows.append((float(line.get('yMin')),' '.join(w.text or '' for w in line.findall('x:word',ns))))
rows.sort()
capstart=next(y for y,t in rows if t.startswith('Figure 1:'))
heading=next(y for y,t in rows if t.replace(' ','')=='ABSTRACT')
caption=[(y,t) for y,t in rows if capstart<=y<heading]
abstract=[(y,t) for y,t in rows if heading<y<740]
assert len(caption)==3,caption
assert len(abstract)==18,abstract
assert abstract[-1][1].endswith('stronger pretrained T2M priors.'),abstract[-1]
assert 720<abstract[-1][0]<725,abstract[-1]
assert re.search(r'\\newlabel\{sec:introduction\}\{\{1\}\{2\}',aux)
report.update(caption_rendered_lines=len(caption),abstract_rendered_lines=len(abstract),abstract_end_page=1,abstract_last_line_top_pt=abstract[-1][0],introduction_start_page=2)


assert re.search(r'\\newlabel\{fig:policy_routes\}\{\{2\}\{2\}',aux)
report['figure_2_on_second_page']=True
page2rows=[]
for line in pages[1].findall('.//x:line',ns):
 if float(line.get('xMin'))>100:
  t=' '.join(w.text or '' for w in line.findall('x:word',ns))
  page2rows.append((float(line.get('yMin')),float(line.get('yMax')),t))
page2rows.sort()
a=next(i for i,(_,_,t) in enumerate(page2rows) if t.startswith('Figure 2:'))
captionrows=page2rows[a:]
b=next(i for i,(_,_,t) in enumerate(captionrows) if t.endswith('generation, respectively.'))
captionrows=captionrows[:b+1]
assert 715<captionrows[-1][1]<738,captionrows
report['figure_2_caption_lines']=len(captionrows)
report['figure_2_caption_bottom_pt']=captionrows[-1][1]
report['figure_2_at_page_bottom']=True
(AUDIT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
