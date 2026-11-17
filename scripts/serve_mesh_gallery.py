"""Read-only mesh gallery; expose only inventoried meshes and viewer assets."""
import argparse,json,struct,subprocess
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from urllib.parse import urlsplit,unquote

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'mesh_screenshot_gallery'
OUT.mkdir(exist_ok=True)
DEPS=ROOT/'runs/new_next_semantic/multiview'
def inventory():
    paths=subprocess.check_output(['rg','--files','runs','-g','*.ply','-g','*.glb'],cwd=ROOT,text=True).splitlines()
    rows=[];errors=[]
    for name in sorted(paths):
        p=ROOT/name
        try:
            with p.open('rb') as f:
                if p.suffix=='.ply':
                    vertices=faces=0
                    for _ in range(150):
                        line=f.readline(10000).decode('ascii').strip()
                        if line.startswith('element vertex '):vertices=int(line.split()[-1])
                        if line.startswith('element face '):faces=int(line.split()[-1])
                        if line=='end_header':break
                    else:raise ValueError('No PLY header terminator')
                    if not faces:continue
                else:
                    magic,version,length=struct.unpack('<III',f.read(12))
                    if magic!=0x46546c67 or version!=2:raise ValueError('Unsupported GLB')
                    size,kind=struct.unpack('<II',f.read(8));data=json.loads(f.read(size))
                    primitives=[q for m in data.get('meshes',[]) for q in m['primitives'] if q.get('mode',4) in (4,5,6)]
                    if not primitives:continue
                    vertices=sum(data['accessors'][q['attributes']['POSITION']]['count'] for q in primitives)
                    faces=sum((data['accessors'][q['indices']]['count'] if 'indices' in q else data['accessors'][q['attributes']['POSITION']]['count'])//3 if q.get('mode',4)==4 else (data['accessors'][q.get('indices',q['attributes']['POSITION'])]['count']-2) for q in primitives)
            rows.append({'id':len(rows),'path':name,'format':p.suffix[1:],'vertices':vertices,'faces':faces,'bytes':p.stat().st_size})
        except Exception as e:errors.append({'path':name,'error':str(e)})
    (OUT/'inventory.json').write_text(json.dumps(rows,indent=2))
    (OUT/'inventory_errors.json').write_text(json.dumps(errors,indent=2))
    return rows

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        route=unquote(urlsplit(self.path).path)
        if route in ('/','/index.html'):p=OUT/'index.html'
        elif route=='/inventory.json':p=OUT/'inventory.json'
        elif route.startswith('/assets/'):
            try:p=ROOT/self.server.rows[int(route.split('/')[2])]['path']
            except (ValueError,IndexError):self.send_error(404);return
        elif route in ['/deps/'+x for x in ['three.module.js','PLYLoader.js','GLTFLoader.js','OrbitControls.js','BufferGeometryUtils.js']]:p=DEPS/route.split('/')[-1]
        else:self.send_error(404);return
        self.asset_path=p
        super().do_GET()
    def translate_path(self,path):return str(self.asset_path)
    def end_headers(self):
        self.send_header('X-Content-Type-Options','nosniff')
        super().end_headers()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--bind',default='0.0.0.0');ap.add_argument('--port',type=int,default=8766);args=ap.parse_args()
    rows=inventory();server=ThreadingHTTPServer((args.bind,args.port),Handler);server.rows=rows
    print(f'Serving {len(rows)} mesh exports at port {args.port}',flush=True)
    server.serve_forever()
