"""Wall-clock scopes, bounded event logging and process-tree resource sampling."""
from collections import defaultdict,deque
from contextlib import contextmanager
import json,os,subprocess,threading,time
from pathlib import Path
import numpy as np


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


class Metrics:
    def __init__(self,output,sample_interval=1.):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.started=time.monotonic();self.samples=defaultdict(lambda:deque(maxlen=8192))
        self.totals=defaultdict(float);self.counts=defaultdict(int);self.lock=threading.Lock()
        self.log=(self.output/'events.jsonl').open('a',buffering=1)
        self.stopped=threading.Event();self.resources={'peak_tree_rss_bytes':0,'peak_cpu_percent':0.,'peak_device_memory_mib':0.,'peak_device_utilization_percent':0.}
        self.thread=threading.Thread(target=self._sample,args=(sample_interval,),daemon=True);self.thread.start()

    def event(self,event,**values):
        with self.lock:self.log.write(json.dumps({'event':event,'wall_seconds':time.monotonic()-self.started,**values},allow_nan=False)+'\n')

    @contextmanager
    def scope(self,name,cuda=False,**metadata):
        if cuda:
            import torch
            if torch.cuda.is_available():torch.cuda.synchronize()
        begin=time.monotonic();error=None
        try:yield
        except BaseException as exc:error=str(exc);raise
        finally:
            if cuda:
                if torch.cuda.is_available():torch.cuda.synchronize()
            elapsed=time.monotonic()-begin
            with self.lock:
                self.samples[name].append(elapsed);self.totals[name]+=elapsed;self.counts[name]+=1
            self.event('stage',stage=name,seconds=elapsed,error=error,**metadata)

    def _sample(self,interval):
        import psutil
        parent=psutil.Process();seen={}
        while not self.stopped.is_set():
            rss=cpu=0.
            try:
                for process in [parent,*parent.children(recursive=True)]:
                    try:
                        if process.pid not in seen:seen[process.pid]=process
                        proc=seen[process.pid];rss+=proc.memory_info().rss;cpu+=proc.cpu_percent()
                    except psutil.Error:pass
                self.resources['peak_tree_rss_bytes']=max(self.resources['peak_tree_rss_bytes'],int(rss))
                self.resources['peak_cpu_percent']=max(self.resources['peak_cpu_percent'],cpu)
                try:
                    data=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=2)
                    memory,util=map(float,data.splitlines()[0].split(','))
                    self.resources['peak_device_memory_mib']=max(self.resources['peak_device_memory_mib'],memory)
                    self.resources['peak_device_utilization_percent']=max(self.resources['peak_device_utilization_percent'],util)
                except (OSError,subprocess.SubprocessError,ValueError,IndexError):pass
            except psutil.Error:pass
            self.stopped.wait(interval)

    def summary(self):
        return {'wall_seconds':time.monotonic()-self.started,'resources':self.resources,
            'resource_scope':'CPU/RSS process tree; GPU samples whole device including other workloads',
            'stages':{name:{'count':self.counts[name],'total_seconds':self.totals[name],
                'mean_seconds':self.totals[name]/self.counts[name],
                'p50_seconds':float(np.percentile(values,50)),'p95_seconds':float(np.percentile(values,95)),
                'percentile_samples':len(values),'percentile_scope':'most recent 8192 calls'} for name,values in self.samples.items()}}

    def close(self):
        self.stopped.set();self.thread.join(timeout=3)
        report=self.summary();save_json(self.output/'timing_summary.json',report);self.log.close();return report
