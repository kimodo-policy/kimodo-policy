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
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
original=json.loads((AUDIT/'source_manifest_before.json').read_text());current={}
for p in SRC.rglob('*'):
 if p.is_file():
  st=p.stat();current[p.relative_to(SRC).as_posix()]={'size':st.st_size,'sha256':digest(p),'mtime_ns':st.st_mtime_ns}
assert current==original,'Original arXiv directory changed'
before=json.loads((checkpoint/'target_manifest_before.json').read_text());expected=json.loads((checkpoint/'expected_edits.json').read_text())
assert set(expected)=={'main.tex','content/Experiments_Results.tex','content/Conclusion.tex','table/simple_single_task.tex','content/Limitations_Future_Work.tex'}
scaling_cp=Path((AUDIT/'latest_scaling_checkpoint.txt').read_text().strip())
scaling_expected=json.loads((scaling_cp/'expected_edits.json').read_text())
expected.update(scaling_expected)
analysis_cp=Path((AUDIT/'latest_transfer_ablation_checkpoint.txt').read_text().strip())
analysis_expected=json.loads((analysis_cp/'expected_edits.json').read_text())
expected.update(analysis_expected)
visual_cp=Path((AUDIT/'latest_visualization_checkpoint.txt').read_text().strip())
visual_expected=json.loads((visual_cp/'expected_edits.json').read_text())
expected.update(visual_expected)
compact_cp=Path((AUDIT/'latest_compact_appendix_checkpoint.txt').read_text().strip())
compact_expected=json.loads((compact_cp/'expected_edits.json').read_text())
expected.update(compact_expected)
for n,h in before.items():assert digest(DST/n)==expected.get(n,h),n
for n,h in expected.items():assert digest(DST/n)==h,n
for name in ['main.tex','content/Experiments_Results.tex','content/Conclusion.tex','table/simple_single_task.tex']:
 s=(checkpoint/name).read_text()
 for a,b in json.loads((checkpoint/(Path(name).stem+'_edits.json')).read_text()):
  assert s.count(a)==1,(name,a[:80]);s=s.replace(a,b)
 extra=scaling_cp/(Path(name).stem+'_edits.json')
 if extra.exists():
  for a,b in json.loads(extra.read_text()):
   assert s.count(a)==1,(name,a[:80]);s=s.replace(a,b)
 extra=analysis_cp/(Path(name).stem+'_edits.json')
 if extra.exists():
  for a,b in json.loads(extra.read_text()):
   assert s.count(a)==1,(name,a[:80]);s=s.replace(a,b)
 extra=visual_cp/(Path(name).stem+'_edits.json')
 if extra.exists():
  for a,b in json.loads(extra.read_text()):
   assert s.count(a)==1,(name,a[:80]);s=s.replace(a,b)
 extra=compact_cp/(Path(name).stem+'_edits.json')
 if extra.exists():
  for a,b in json.loads(extra.read_text()):
   assert s.count(a)==1,(name,a[:80]);s=s.replace(a,b)
 assert s==(DST/name).read_text(),name
oldcon=(checkpoint/'content/Conclusion.tex').read_text();app=(DST/'content/Limitations_Future_Work.tex').read_text()
assert app.split('\\textbf{Limitations.}',1)[1]==oldcon.split('\\textbf{Limitations.}',1)[1]
assert app.count('\\section{')==1 and '\\label{sec:limitations_future}' in app
newcon=(DST/'content/Conclusion.tex').read_text()
assert len([p for p in newcon.split('\n\n') if p.strip() and not p.startswith('\\section')])==1
assert '\\ref{sec:limitations_future}' in newcon
oldt=(checkpoint/'table/simple_single_task.tex').read_text();newt=(DST/'table/simple_single_task.tex').read_text()
a=oldt.index('    ACT~');b=oldt.index('    $\\Psi_0$',a)
assert newt==oldt[:a]+oldt[b:]
removed=oldt[a:b]
assert newt.count('\\cite{')==1 and '\\cite{psi0}' in newt
for n in ['table/arena_single_task.tex','table/arena_multi_task.tex','table/proof_t2m.tex']:
 assert digest(DST/n)==before[n]
oldexp=(checkpoint/'content/Experiments_Results.tex').read_text();newexp=(DST/'content/Experiments_Results.tex').read_text()
nums=lambda t:set(re.findall(r'\d+(?:\.\d+)?',re.sub(r'\\[A-Za-z@]+',' ',t)))
assert nums(oldexp)==nums(newexp),(nums(oldexp)-nums(newexp),nums(newexp)-nums(oldexp))
assert newexp.count('\\textbf{Results Analysis.}')==1
ab=newexp.split('\\textbf{Results Analysis.}',1)[1].split('\\endgroup',1)[0].strip()
assert '\n\n' not in ab
assert '\\textbf{Effect of' not in newexp
assert (checkpoint/'content/Method.tex').read_bytes()==(DST/'content/Method.tex').read_bytes()
def flatten(name,old=False):
 p=checkpoint/name if old and (checkpoint/name).is_file() else DST/name
 if old and name=='content/Task_Visualizations.tex':p=visual_cp/name
 s=re.sub(r'(?<!\\)%[^\n]*','',p.read_text())
 return re.sub(r'\\(?:input|include)\{([^}]+)\}',lambda m:flatten(m[1] if Path(m[1]).suffix else m[1]+'.tex',old),s)
body=flatten('main.tex').split('\\begin{document}',1)[1];oldbody=flatten('main.tex',True).split('\\begin{document}',1)[1]
assert Counter(re.findall(r'\\cite\{([^}]+)\}',body))==Counter(re.findall(r'\\cite\{([^}]+)\}',oldbody))-Counter(re.findall(r'\\cite\{([^}]+)\}',removed))
assert re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',body,re.S)==re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',oldbody,re.S)
report={'source_unchanged_files':len(current),'edited_tex_files':list(expected),'only_table3_baseline_rows_removed':True,'table3_retained_rows':['Psi_0','Ours*'],'other_tables_unchanged':True,'all_experiment_numerical_values_preserved':True,'limitations_future_moved_verbatim':True,'ablation_results_one_paragraph':True,'conclusion_one_paragraph':True,'method_unchanged':True,'equations_unchanged':True,'citations_removed_only_with_deleted_baseline_rows':True}
for kind in ['figure','table','equation']:
 pattern=r'\\begin\{'+kind+r'\*?\}';assert len(re.findall(pattern,body))==len(re.findall(pattern,oldbody));report[kind+'_count']=len(re.findall(pattern,body))
for name,pattern,added in [('labels',r'\\label\{([^}]+)\}',['sec:limitations_future','sec:scaling_analysis_appendix','sec:t2m_analysis_appendix','sec:ablation_analysis_appendix']),('reference_targets',r'\\(?:ref|eqref|autoref)\{([^}]+)\}',['sec:limitations_future','sec:scaling_analysis_appendix','fig:scaling_analysis','sec:t2m_analysis_appendix','sec:ablation_analysis_appendix','tab:t2m_dependency','tab:t2m_dependency','fig:training_loss_comparison','fig:humanoidarena','sec:exp_simulation','fig:simple_tasks','sec:exp_simple','fig:real_world','sec:exp_real'])]:
 assert Counter(re.findall(pattern,body))==Counter(re.findall(pattern,oldbody))+Counter(added),name
 report[name]=len(re.findall(pattern,body))
for img in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',body):
 assert (DST/img).is_file() and digest(DST/img)==before[img],img
with zipfile.ZipFile(DST/'iclr-2027-style-files.zip') as z:
 for n in ['iclr2027_conference.sty','iclr2027_conference.bst','natbib.sty','fancyhdr.sty','math_commands.tex']:assert (DST/n).read_bytes()==z.read('iclr2027/'+n)
report['official_template_and_images_unchanged']=True
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
for name in ['Dual-Path paragraph','HandNet paragraph']:
 assert layout[name]['lines']==8 and layout[name]['pages']==[5],(name,layout[name])
assert layout['Figure 4 caption paragraph']['lines']==3
assert layout['Single-task']['pages']==[6] and layout['Single-task']['lines']==8
assert position('tab:arena_single_task')['page']==6
assert layout['Experimental Results opening']['lines']==4 and layout['HumanoidArena opening']['lines']==3
assert position('sec:conclusion')['page']==9
assert position('sec:limitations_future')['number']=='G'
assert layout['Conclusion paragraph']['pages']==[9]
conrows=[r for r in allrows if r[0]==9 and r[2].endswith('human T2M backbones.')]
assert conrows and 720<conrows[-1][1]<725,conrows
assert 'REFERENCES' in ''.join(''.join(w.text or '' for w in l.findall('x:word',ns)) for l in pages[9].findall('.//x:line',ns)).upper()
assert not report['overfull_box_warnings']
report['main_text_ends_page']=9
report['main_text_last_line_top_pt']=conrows[-1][1]
report['references_start_page']=10
report['limitations_future_appendix']=position('sec:limitations_future')
report['scaling_analysis_appendix']=position('sec:scaling_analysis_appendix')
assert report['scaling_analysis_appendix']['number']=='E'
assert layout['Scaling results']['pages']==[9]
report['scaling_results_one_paragraph']=True
report['t2m_analysis_appendix']=position('sec:t2m_analysis_appendix')
report['ablation_analysis_appendix']=position('sec:ablation_analysis_appendix')
assert report['t2m_analysis_appendix']['number']=='D'
assert report['ablation_analysis_appendix']['number']=='F'
report['paragraph_layout']=layout
report['first_six_page_requirements_preserved']=True
(AUDIT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
