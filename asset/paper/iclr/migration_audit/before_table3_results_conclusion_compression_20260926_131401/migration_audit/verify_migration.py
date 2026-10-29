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
assert set(expected)=={'content/Experiments_Results.tex'}
for name,sha in expected.items(): assert digest(DST/name)==sha,name
assert '\\input{content/Teaser}\n\\input{content/Abstract}\n\\clearpage' in (DST/'main.tex').read_text()
def flatten(name,old=False):
 p=checkpoint/name if old and (checkpoint/name).is_file() else DST/name
 s=re.sub(r'(?<!\\)%[^\n]*','',p.read_text())
 return re.sub(r'\\(?:input|include)\{([^}]+)\}',lambda m:flatten(m[1] if Path(m[1]).suffix else m[1]+'.tex',old),s)
body=flatten('main.tex').split('\\begin{document}',1)[1]
oldbody=flatten('main.tex',True).split('\\begin{document}',1)[1]
report={'source_unchanged_files':len(current),'edited_tex_files':list(expected),'other_current_project_files_unchanged':True}







newintro=(DST/'content/Introduction.tex').read_text()
oldmethod=(checkpoint/'content/Method.tex').read_text()
newmethod=(DST/'content/Method.tex').read_text()
expected_method=oldmethod
for old,new in json.loads((checkpoint/'method_prose_edits.json').read_text()):
 assert expected_method.count(old)==1,old[:80]
 expected_method=expected_method.replace(old,new)
assert expected_method==newmethod
oldexp=(checkpoint/'content/Experiments_Results.tex').read_text()
expected_exp=oldexp
for old,new in json.loads((checkpoint/'experiments_prose_edits.json').read_text())+json.loads((checkpoint/'layout_edits.json').read_text()):
 assert expected_exp.count(old)==1
 expected_exp=expected_exp.replace(old,new)
assert expected_exp==(DST/'content/Experiments_Results.tex').read_text()
eqpattern=r'\\begin\{equation\}.*?\\end\{equation\}'
assert re.findall(eqpattern,oldmethod,re.S)==re.findall(eqpattern,newmethod,re.S)
assert Counter(re.findall(r'\\cite\{([^}]+)\}',body))==Counter(re.findall(r'\\cite\{([^}]+)\}',oldbody))
report['changes_limited_to_two_experiment_opening_paragraphs']=True
report['equations_unchanged']=True
assert oldexp.split('\\textbf{Single-task results.}',1)[1]==(DST/'content/Experiments_Results.tex').read_text().split('\\textbf{Single-task results.}',1)[1], 'Result text or figures changed'
report['all_experiment_results_captions_and_tables_unchanged']=True
report['citations_preserved']=True
for kind in ['figure','table','equation']:
 pattern=r'\\begin\{'+kind+r'\*?\}'
 assert len(re.findall(pattern,body))==len(re.findall(pattern,oldbody))
 report[kind+'_count']=len(re.findall(pattern,body))


for name,pattern,removed in [
 ('labels',r'\\label\{([^}]+)\}',[]),
 ('reference_targets',r'\\(?:ref|eqref|autoref)\{([^}]+)\}',[])
]:
 added=[]
 assert Counter(re.findall(pattern,body))==Counter(re.findall(pattern,oldbody))-Counter(removed)+Counter(added)
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

page3rows=[]
for line in pages[2].findall('.//x:line',ns):
 x,y=float(line.get('xMin')),float(line.get('yMin'))
 if x>100 and 80<y<740:
  page3rows.append((y,float(line.get('yMax')),' '.join(w.text or '' for w in line.findall('x:word',ns))))
page3rows.sort()
a=next(i for i,(_,_,t) in enumerate(page3rows) if t.startswith('T2M includes'))
t2mrows=page3rows[a:]
assert len(t2mrows)==13,t2mrows
assert t2mrows[-1][2].endswith('distinct control requirements.'),t2mrows[-1]
assert 718<t2mrows[-1][1]<737,t2mrows[-1]
report['t2m_rendered_lines']=len(t2mrows)
report['t2m_end_page']=3
report['t2m_last_line_bottom_pt']=t2mrows[-1][1]
def position(label):
 m=re.search(r'\\newlabel\{'+re.escape(label)+r'\}\{\{([^}]*)\}\{(\d+)\}',aux)
 assert m,label
 return {'number':m[1],'page':int(m[2])}
assert position('sec:task_visualizations')['number']=='C'
report['task_visualization_appendix']=position('sec:task_visualizations')
report['moved_visualizations']={}
for fig,sec in [('fig:humanoidarena','sec:vis_humanoidarena'),('fig:simple_tasks','sec:vis_simple'),('fig:real_world','sec:vis_real_world')]:
 fp,sp=position(fig),position(sec)
 assert sp['number'].startswith('C.') and fp['page']>=sp['page']
 assert fp['page']>=position('sec:task_visualizations')['page']
 report['moved_visualizations'][fig]={'figure':fp, 'appendix_section':sp}


assert position('fig:method_architecture')['number']=='3'
assert position('fig:method_architecture')['page']==4
assert position('sec:method')['page']==4
assert 'eq:three_stage_architecture' not in aux
assert position('eq:policy_training_objective')['number']=='2'
page4rows=[]
for line in pages[3].findall('.//x:line',ns):
 x,y=float(line.get('xMin')),float(line.get('yMin'))
 if x>100 and 80<y<740:
  page4rows.append((y,float(line.get('yMax')),' '.join(w.text or '' for w in line.findall('x:word',ns))))
page4rows.sort()
head=next(y for y,_,t in page4rows if 'METHOD:KIMODO-POLICY' in t.replace(' ','').upper())
modelhead=next(y for y,_,t in page4rows if 'MODELARCHITECTURE' in t.replace(' ','').upper())
capstart=next(y for y,_,t in page4rows if t.startswith('Figure 3:'))
assert head<capstart<modelhead,(head,capstart,modelhead)
figcaption=[r for r in page4rows if capstart<=r[0]<modelhead]
caption_line_y=[]
for row in figcaption:
 if not any(abs(row[0]-y)<1 for y in caption_line_y): caption_line_y.append(row[0])
assert len(caption_line_y)<=6,figcaption
report['figure_3_page']=4
report['figure_3_between_method_opening_and_section_3_1']=True
report['figure_3_caption_rendered_lines']=len(caption_line_y)
report['method_section_heading_top_pt']=head
allrows=[]
for page_no,page in enumerate(pages,1):
 rows=[]
 for line in page.findall('.//x:line',ns):
  x,y=float(line.get('xMin')),float(line.get('yMin'))
  if x>100 and 80<y<740:
   rows.append((page_no,y,' '.join(w.text or '' for w in line.findall('x:word',ns))))
 allrows.extend(sorted(rows))
a=next(i for i,r in enumerate(allrows) if r[2].startswith('Motion-History Conditioning.'))
b=next(i for i in range(a+1,len(allrows)) if 'TRAININGANDINFERENCE' in allrows[i][2].replace(' ','').upper())
historyrows=[r for r in allrows[a:b] if not re.fullmatch(r'[0-9.]+',r[2])]
assert len(historyrows)==9,historyrows
assert set(r[0] for r in historyrows)=={5},historyrows
assert 722<historyrows[-1][1]<725,historyrows[-1]
assert position('sec:training_inference')['page']==6
assert historyrows[-1][2].endswith('state-based correction.'),historyrows[-1]
report['motion_history_rendered_lines']=len(historyrows)
report['motion_history_pages']=sorted(set(r[0] for r in historyrows))
report['motion_history_last_line_top_pt']=historyrows[-1][1]
report['motion_history_ends_at_page5_bottom']=True
report['remaining_equations_renumbered']=True
subprocess.run(['python3',str(AUDIT/'check_paragraph_layout.py')],check=True,capture_output=True,text=True)
layout=json.loads((AUDIT/'paragraph_layout.json').read_text())
for name in ['Three-Stage','Dual-Path','HandNet','Training lead','Training ending','Inference','Figure 4 caption']:
 assert layout[name]['last_width_pt']>150,(name,layout[name])
for name in ['Single-task','Multi-task']:
 assert layout[name]['lines']<=9,(name,layout[name])
 assert layout[name]['last_width_pt']>120,(name,layout[name])
for name,lines in [('SIMPLE results',8),('Table 3 caption',3),('T2M setup',5),('T2M transfer',8),('Scaling setup',5)]:
 assert layout[name]['lines']==lines,(name,layout[name])
 assert layout[name]['last_width_pt']>150,(name,layout[name])
for name,lines in [('Table 1 caption',3),('Table 2 caption',1),('Figure 5 caption',3)]:
 assert layout[name]['lines']==lines,(name,layout[name])
assert layout['Single-task']['pages']==[6],layout['Single-task']
assert layout['Single-task']['lines']==8,layout['Single-task']
assert layout['Attention analysis']['lines']<=6
report['single_task_results_fully_on_page6']=True
report['experiment_positions']={label:position(label) for label in ['tab:arena_single_task','tab:arena_multi_task','fig:attention','sec:exp_simple','sec:exp_real','fig:real_world_success_rates','sec:exp_t2m','sec:exp_scaling','fig:scaling_analysis','sec:exp_ablation','fig:training_loss_comparison','sec:conclusion']}
assert position('tab:arena_single_task')['page']==6
assert layout['Experimental Results opening']['lines']==4
assert layout['HumanoidArena opening']['lines']==3
report['table1_and_single_task_results_on_page6']=True
report['opening_paragraphs_three_lines_shorter']=True
for name in ['Dual-Path paragraph','HandNet paragraph']:
 assert layout[name]['lines']==8,(name,layout[name])
 assert layout[name]['pages']==[5],(name,layout[name])
report['dual_path_and_handnet_each_8_lines']=True
assert layout['Figure 4 caption paragraph']['lines']==3
assert layout['Figure 4 caption paragraph']['pages']==[5]
assert layout['Motion Prior Transfer paragraph']['lines']==9
report['figure4_caption_three_lines']=True
report['paragraph_layout']=layout
report['method_short_terminal_lines_resolved']=True
(AUDIT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
