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
assert set(expected)=={'content/Introduction.tex','content/Related_Work.tex'}
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
pattern=r'\\begin\{itemize\}.*?\\end\{itemize\}'
a=re.search(pattern,oldintro,re.S); b=re.search(pattern,newintro,re.S)
assert a and b
assert oldintro[:a.start()]+oldintro[a.end():]==newintro[:b.start()]+newintro[b.end():]
assert len(re.findall(r'\\item\b',a.group()))==len(re.findall(r'\\item\b',b.group()))==3
oldrw=(checkpoint/'content/Related_Work.tex').read_text()
newrw=(DST/'content/Related_Work.tex').read_text()
assert re.findall(r'\\cite\{([^}]+)\}',oldrw)==re.findall(r'\\cite\{([^}]+)\}',newrw)
for oldstart,newstart,ending in [
 ('Humanoid vision--language--action','Humanoid VLA models',r'.*?(?=\n\s*\n)'),
 ('Text-to-motion (T2M)','T2M generation evolved',r'.*')
]:
 a=re.search(r'^'+re.escape(oldstart)+ending,oldrw,re.M|re.S)
 b=re.search(r'^'+re.escape(newstart)+ending,newrw,re.M|re.S)
 assert a and b
 oldrw=oldrw[:a.start()]+'PARAGRAPH_REVISED'+oldrw[a.end():]
 newrw=newrw[:b.start()]+'PARAGRAPH_REVISED'+newrw[b.end():]
assert oldrw==newrw
report['changes_limited_to_related_work_and_contribution_text']=True
report['related_work_citations_preserved_in_order']=True
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

introlines=[]
for line in pages[1].findall('.//x:line',ns):
 if float(line.get('xMin'))>100:
  introlines.append((float(line.get('yMin')),' '.join(w.text or '' for w in line.findall('x:word',ns))))
introlines.sort()
starts=['Vision','Humanoid policy learning','Despite these advances','Recent text-to-motion']
indices=[next(i for i,(_,t) in enumerate(introlines) if t.startswith(start)) for start in starts]
counts=[indices[i+1]-indices[i] for i in range(3)]
assert all(9<=n<=11 for n in counts),counts
report['intro_first_three_paragraph_rendered_lines']=counts

p4_end=next(i for i,(_,t) in enumerate(introlines) if 'We introduce Kimodo-Policy' in t)
p4_count=p4_end-indices[3]
assert 12<=p4_count<=15,p4_count
report['intro_fourth_paragraph_rendered_lines']=p4_count

allblocks=[]
allrows=[]
for page_index,page in enumerate(pages,1):
 page_rows=[]
 for block in page.findall('.//x:block',ns):
  bl=[]
  for line in block.findall('x:line',ns):
   x,y=float(line.get('xMin')),float(line.get('yMin'))
   if x>100 and 80<y<740:
    t=' '.join(w.text or '' for w in line.findall('x:word',ns))
    bl.append({'page':page_index,'y':y,'xmax':float(line.get('xMax')),'text':t})
  if bl:allblocks.append(bl);page_rows.extend(bl)
 allrows.extend(sorted(page_rows,key=lambda a:a['y']))
rwcounts=[]
for start,end in [('Humanoid VLA models typically','whole-body behavior.'),('T2M generation evolved','control requirements.')]:
 a=next(i for i,r in enumerate(allrows) if r['text'].startswith(start))
 b=next(i for i in range(a,len(allrows)) if allrows[i]['text'].endswith(end))
 rwcounts.append(b-a+1)
assert rwcounts==[15,15],rwcounts
contribs=[]
for start in ['We introduce Kimodo-Policy','We propose dedicated','We demonstrate strong performance']:
 block=next(b for b in allblocks if any(r['page']==2 and r['text'].startswith('• '+start) for r in b))
 offset=next(i for i,r in enumerate(block) if start in r['text'])
 block=block[offset:]
 stop=next((i for i,r in enumerate(block[1:],1) if any(t in r['text'] for t in ['We introduce Kimodo-Policy','We propose dedicated','We demonstrate strong performance'])),len(block))
 block=block[:stop]
 assert len(block)==3,(start,len(block))
 fill=(block[-1]['xmax']-144)/360
 assert .90<fill<=1.001,fill
 contribs.append({'lines':len(block),'last_line_fill_percent':round(fill*100,1)})
report['related_work_paragraph_rendered_lines']=rwcounts
report['contribution_layout']=contribs




(AUDIT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
