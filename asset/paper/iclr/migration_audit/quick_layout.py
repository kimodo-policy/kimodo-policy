from pathlib import Path
import subprocess,shutil,re,json,xml.etree.ElementTree as E
root=Path.cwd();out=root/'migration_audit/layout_preview';out.mkdir(exist_ok=True)
for name in ['main.aux','main.bbl']:
 if (root/name).exists(): shutil.copy2(root/name,out/name)
for run in range(2):
 with (out/('build_'+str(run)+'.log')).open('w') as f:
  subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-jobname=main','-output-directory='+str(out),r'\PassOptionsToPackage{draft}{graphicx}\input{main.tex}'],stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run(['pdftotext','-bbox-layout',str(out/'main.pdf'),str(out/'bbox.html')],check=True)
ns={'x':'http://www.w3.org/1999/xhtml'};rows=[]
for pn,page in enumerate(E.parse(out/'bbox.html').getroot().findall('.//x:page',ns),1):
 groups=[]
 parts=sorted((float(l.get('yMin')),float(l.get('xMin')),float(l.get('xMax')),' '.join(w.text or '' for w in l.findall('x:word',ns))) for l in page.findall('.//x:line',ns) if float(l.get('xMin'))>100 and 80<float(l.get('yMin'))<740)
 for y,x,xmax,t in parts:
  if groups and abs(y-groups[-1]['y'])<1:
   g=groups[-1];g['parts'].append((x,t));g['xmin']=min(g['xmin'],x);g['xmax']=max(g['xmax'],xmax)
  else: groups.append({'page':pn,'y':y,'xmin':x,'xmax':xmax,'parts':[(x,t)]})
 for g in groups:
  g['text']=' '.join(t for x,t in sorted(g.pop('parts')));g['width']=round(g['xmax']-g['xmin'],1);rows.append(g)
report={}
for name,end in [('Dual-Path Fusion.','conditioning.'),('HandNet.','ControlNet.')]:
 a=next(i for i,r in enumerate(rows) if r['text'].startswith(name));b=next(i for i in range(a,len(rows)) if rows[i]['text'].endswith(end));part=rows[a:b+1]
 report[name]={'lines':len(part),'pages':sorted(set(r['page'] for r in part)),'last_width':part[-1]['width'],'text':[r['text'] for r in part]}
aux=(out/'main.aux').read_text();report['positions']={}
for label in ['fig:method_architecture','sec:method_architecture','fig:t2m_transfer','sec:training_inference','tab:arena_single_task','tab:arena_multi_task','fig:attention','sec:exp_simple']:
 m=re.search(r'\\newlabel\{'+re.escape(label)+r'\}\{\{([^}]*)\}\{([^}]*)\}',aux);report['positions'][label]=m.groups() if m else None
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
