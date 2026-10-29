from pathlib import Path
import subprocess,json,xml.etree.ElementTree as E
root=Path.cwd();bbox=root/'migration_audit/main_bbox.html'
subprocess.run(['pdftotext','-bbox-layout',str(root/'main.pdf'),str(bbox)],check=True)
ns={'x':'http://www.w3.org/1999/xhtml'};pages=E.parse(bbox).getroot().findall('.//x:page',ns);rows=[]
for pn,page in enumerate(pages,1):
 parts=sorted((float(l.get('yMin')),float(l.get('xMin')),float(l.get('xMax')),' '.join(w.text or '' for w in l.findall('x:word',ns))) for l in page.findall('.//x:line',ns) if float(l.get('xMin'))>100 and 80<float(l.get('yMin'))<740)
 groups=[]
 for y,x,xmax,t in parts:
  if groups and abs(y-groups[-1]['y'])<1:
   g=groups[-1];g['parts'].append((x,t));g['xmin']=min(g['xmin'],x);g['xmax']=max(g['xmax'],xmax)
  else: groups.append({'page':pn,'y':y,'xmin':x,'xmax':xmax,'parts':[(x,t)]})
 for g in groups:
  g['text']=' '.join(t for x,t in sorted(g.pop('parts')));g['width']=round(g['xmax']-g['xmin'],1);rows.append(g)
report={}
for name,start,suffix in [('Three-Stage','Three-Stage Architecture.','commands.'),('Dual-Path','Dual-Path Fusion.','conditioning.'),('HandNet','HandNet.','ControlNet.'),('Training lead','Training Objective.','frames:'),('Training ending','Training Objective.','frozen.'),('Inference','Inference.','correction.'),('Figure 4 caption','Figure 4:','motion.')]:
 a=next(i for i,r in enumerate(rows) if 4<=r['page']<=6 and r['text'].startswith(start))
 i=next(i for i in range(a,len(rows)) if rows[i]['text'].endswith(suffix));row=rows[i]
 report[name]={'last_width_pt':row['width'],'last_two_lines':[r['text'] for r in rows[i-1:i+1]],'page':row['page']}
for name,table in [('Single-task',1),('Multi-task',2)]:
 a=next(i for i,r in enumerate(rows) if r['text'].startswith(name+' results.'))
 b=next(i for i in range(a+1,len(rows)) if rows[i]['text'].startswith('Table '+str(table)+':'))
 para=rows[a:b];report[name]={'lines':len(para),'pages':sorted(set(r['page'] for r in para)),'last_width_pt':para[-1]['width'],'text':[r['text'] for r in para]}
def span(start,end,occurrence=0):
 hits=[i for i,r in enumerate(rows) if r['text'].startswith(start)]
 a=hits[occurrence]
 b=next(i for i in range(a,len(rows)) if rows[i]['text'].endswith(end))
 selected=rows[a:b+1]
 return {'lines':len(selected),'pages':sorted(set(r['page'] for r in selected)),'last_width_pt':selected[-1]['width'],'text':[r['text'] for r in selected]}
report['SIMPLE results']=span('Single-task results.','target-pose variation.',1)
report['Table 3 caption']=span('Table 3:','aggregate rates.')
report['T2M setup']=span('Kimodo-Policy uses the pretrained Kimodo','Rigplay (Rempe et al., 2026).')
report['T2M transfer']=span('T2M transfer results.','trade-offs.')
report['Scaling setup']=span('We isolate data and capacity effects','evaluation suite.')
report['Table 1 caption']=span('Table 1:','results.')
report['Table 2 caption']=span('Table 2:','tasks.')
report['Figure 5 caption']=span('Figure 5:','regions.')
report['Attention analysis']=span('Attention map analysis.','selection.')
report['Dual-Path paragraph']=span('Dual-Path Fusion.','conditioning.')
report['HandNet paragraph']=span('HandNet.','ControlNet.')
report['Training prose after equation']=span('The masked MSE terms','frozen.')
report['Inference paragraph']=span('Inference.','correction.')
report['Figure 4 caption paragraph']=span('Figure 4:','motion.')
report['Motion Prior Transfer paragraph']=span('Motion Prior Transfer.','execution.')
report['Motion-History Conditioning paragraph']=span('Motion-History Conditioning.','state-based correction.')
report['Experimental Results opening']=span('We evaluate Kimodo-Policy through','evaluation environments.')
report['HumanoidArena opening']=span('HumanoidArena (','visualizes the tasks.')
(root/'migration_audit/paragraph_layout.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
