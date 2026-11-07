"""Bounded asynchronous JSONL recorder; no robot commands or control decisions."""
import json
import queue
import threading
import time
from pathlib import Path
import numpy as np


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [plain(v) for v in value]
    if isinstance(value, BaseException): return str(value)
    return value


class Recorder:
    def __init__(self, directory=None):
        self.dropped=0; self.error=None; self.queue=None; self.thread=None
        if directory:
            path=Path(directory).expanduser(); path.mkdir(parents=True,exist_ok=True)
            self.directory=path
            self.file=(path/'events.jsonl').open('x',encoding='utf-8')
            self.queue=queue.Queue(maxsize=64)
            self.thread=threading.Thread(target=self._write,daemon=True); self.thread.start()
    def emit(self,event,**data):
        if self.queue is None: return
        row=plain(dict(event=event,monotonic=time.monotonic(),wall_time=time.time(),**data))
        try:self.queue.put_nowait(row)
        except queue.Full:self.dropped+=1
    def image(self,name,pixels):
        if self.queue is None: return
        try:self.queue.put_nowait((name,np.asarray(pixels).copy()))
        except queue.Full:self.dropped+=1
    def _write(self):
        try:
            while True:
                row=self.queue.get()
                if row is None:break
                if isinstance(row,tuple):
                    from PIL import Image
                    dest=self.directory/'images';dest.mkdir(exist_ok=True)
                    Image.fromarray(row[1]).save(dest/row[0],quality=90)
                    continue
                self.file.write(json.dumps(row,ensure_ascii=False)+'\n')
            self.file.write(json.dumps(dict(event='recorder_end',dropped=self.dropped))+'\n')
        except Exception as e:
            self.error=str(e);print('[diagnostics] write failed: '+str(e),flush=True)
        finally:self.file.close()
    def close(self):
        if self.thread is None:return
        if self.thread.is_alive():
            try:self.queue.put(None,timeout=2)
            except queue.Full:
                print('[diagnostics] shutdown queue full',flush=True);return
            self.thread.join(timeout=5)
        if self.thread.is_alive() or self.dropped or self.error:
            print(f'[diagnostics] incomplete: dropped={self.dropped} error={self.error}',flush=True)
