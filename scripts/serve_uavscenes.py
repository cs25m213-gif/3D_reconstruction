#!/usr/bin/env python3
"""Serve the UAVScenes batch and new_next_semantic through numbered routes."""

import argparse
from functools import partial
import html
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


PROJECT = Path(__file__).resolve().parents[1]
BATCH = PROJECT / "runs/uavscenes_30_batch"
EXTRA = PROJECT / "runs/new_next_semantic"
STYLE = """
*{box-sizing:border-box}body{margin:0;background:#0e151c;color:#e2ebf3;font:15px system-ui}
a{color:#83d8ff}header{padding:14px 20px;background:#17222c;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
main{max-width:1100px;margin:30px auto;padding:0 20px}h1{font-size:28px}
input,select{background:#233545;color:#e2ebf3;border:1px solid #456077;padding:9px;border-radius:5px}
input{width:100%;margin:12px 0}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:12px;border-bottom:1px solid #304252}
small{color:#a6bed1}.viewer{height:100dvh;display:flex;flex-direction:column}.viewer iframe{flex:1;min-height:0;width:100%;border:0}
.viewer{overflow:hidden}.viewer>header{height:44px;min-height:44px;padding:5px 12px;gap:12px;flex-wrap:nowrap;font-size:13px;overflow-x:auto}
.viewer>header>*{flex-shrink:0;white-space:nowrap}.viewer>header>b{flex-shrink:1;min-width:80px;overflow:hidden;text-overflow:ellipsis}
.viewer select{padding:5px;max-width:170px}.viewer iframe{display:block}
"""


def compact_viewer(page):
    """Adapt generated mesh pages at serving time, keeping reconstruction outputs intact."""
    style = """<style id="compact-viewer-style">
html,body{height:100%;height:100dvh;overflow:hidden}
body{display:flex;flex-direction:column}
header{height:40px;min-height:40px;flex:0 0 40px;padding:4px 10px;gap:10px;flex-wrap:nowrap;position:relative;z-index:2}
#view{height:auto;flex:1 1 0;min-height:0;overflow:hidden}
#status{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px}
#overview{order:-1;white-space:nowrap}
#viewer-tools{flex-shrink:0;font-size:13px}
#viewer-tools>summary{cursor:pointer;padding:5px 9px;border:1px solid #456077;border-radius:4px;user-select:none}
#viewer-tool-panel{position:absolute;right:8px;top:39px;width:min(520px,calc(100vw - 16px));max-height:calc(100dvh - 56px);overflow:auto;padding:16px;display:flex;flex-wrap:wrap;align-items:center;gap:14px;background:#17222c;border:1px solid #456077;border-radius:6px;box-shadow:0 10px 30px #0008}
#viewer-tools:not([open])>#viewer-tool-panel{display:none}
#viewer-tool-panel>b,#viewer-tool-panel>small,#viewer-tool-panel>#summary{flex-basis:100%}
#viewer-tool-panel>small{line-height:1.5}
</style>"""
    script = """<script>
(()=>{
 const header=document.querySelector('header');
 const tools=document.createElement('details');tools.id='viewer-tools';
 const label=document.createElement('summary');label.textContent='Controls & info';
 const panel=document.createElement('div');panel.id='viewer-tool-panel';
 tools.append(label,panel);
 const keep=new Set(['overview','status','viewer-tools']);
 function collect(){for(const item of [...header.children])if(!keep.has(item.id))panel.append(item)}
 collect();header.append(tools);
 document.getElementById('overview').textContent='Fit model';
 new MutationObserver(collect).observe(header,{childList:true});
 // Resize WebGL after layout changes, including iframe and browser resizing.
 new ResizeObserver(()=>window.dispatchEvent(new Event('resize'))).observe(document.getElementById('view'));
 document.addEventListener('keydown',e=>{if(e.key==='Escape')tools.open=false});
 document.addEventListener('pointerdown',e=>{if(!tools.contains(e.target))tools.open=false});
})();
</script>"""
    # Fit the bounding sphere using the narrower field of view, including portrait windows.
    old_overview = "function overview(){camera.up.set(0,1,0);camera.fov=55;camera.position.set(radius,radius*.7,radius);controls.target.set(0,0,0);camera.updateProjectionMatrix();controls.update()}"
    new_overview = """function overview(){
 camera.up.set(0,1,0);camera.fov=55;
 const halfVertical=THREE.MathUtils.degToRad(camera.fov/2);
 const halfHorizontal=Math.atan(Math.tan(halfVertical)*camera.aspect);
 const sphereRadius=Math.max(box.getSize(new THREE.Vector3()).length()/2,.001);
 const distance=1.08*sphereRadius/Math.sin(Math.min(halfVertical,halfHorizontal));
 camera.far=Math.max(radius*100,distance+sphereRadius*2);
 camera.position.set(1,.7,1).normalize().multiplyScalar(distance);
 controls.target.set(0,0,0);camera.updateProjectionMatrix();controls.update();
 }"""
    page = page.replace(old_overview, new_overview)
    page = page.replace('preview?captureView():overview();', 'overview();')
    page = page.replace('</style>', '</style>' + style, 1)
    return page.replace('<script type="module">', script + '<script type="module">', 1)


def scene_name(scene):
    return scene.parent.name if scene.name == "reconstruction" else scene.name


class BatchHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, scenes, **kwargs):
        self.scenes = scenes
        super().__init__(*args, **kwargs)

    def page(self, title, body, body_class=""):
        data = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{html.escape(title)}</title><style>{STYLE}</style></head>'
                f'<body class="{body_class}">{body}</body></html>').encode()
        return self.respond(data, "text/html; charset=utf-8")

    def respond(self, data, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        return io.BytesIO(data)

    def send_head(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/routes.json":
            routes = [{"id": i, "scene": scene_name(scene), "url": f"/{i}"}
                      for i, scene in enumerate(self.scenes, 1)]
            return self.respond(json.dumps(routes, indent=2).encode(), "application/json")
        if path == "/":
            rows = "".join(
                f'<tr><td><a href="/{i}">/{i}</a></td>'
                f'<td><a href="/{i}">{html.escape(scene_name(scene))}</a></td>'
                f'<td><a href="/{i}/multiview/scene_colored.glb" download>GLB</a> · '
                f'<a href="/{i}/multiview/scene_mesh_colored.ply" download>PLY</a></td></tr>'
                for i, scene in enumerate(self.scenes, 1))
            return self.page("3D reconstruction models", f"""
                <main><h1>3D reconstructions · {len(self.scenes)} models</h1>
                <p>Open a numbered route to explore its reconstruction.</p>
                <small>UAVScenes batch in alphabetical order, followed by new_next_semantic</small>
                <input id="search" aria-label="Filter scenes" placeholder="Filter by scene name or route…">
                <table><thead><tr><th>Route</th><th>Scene</th><th>Download</th></tr></thead><tbody>{rows}</tbody></table>
                </main><script>document.getElementById('search').oninput=e=>{{
                for(const row of document.querySelectorAll('tbody tr'))
                row.hidden=!row.textContent.toLowerCase().includes(e.target.value.toLowerCase());
                }};</script>""")
        match = re.fullmatch(r"/([1-9][0-9]*)(?:/(.*))?", path)
        if not match or not 1 <= int(match[1]) <= len(self.scenes):
            self.send_error(404, "Unknown model route")
            return None
        number = int(match[1])
        scene = self.scenes[number - 1]
        relative = match[2]
        if not relative:
            previous = f'<a href="/{number - 1}">← Previous</a>' if number > 1 else ""
            following = f'<a href="/{number + 1}">Next →</a>' if number < len(self.scenes) else ""
            choices = [("multiview/mesh_viewer.html", "Colored mesh"),
                       ("multiview/supported_viewer.html", "Supported mesh"),
                       ("sparse_run/online/viewer.html", "Sparse point cloud"),
                       ("multiview/coverage.html", "Coverage comparison")]
            options = "".join(f'<option value="/{number}/{p}">{label}</option>'
                              for p, label in choices if (scene / p).is_file())
            return self.page(scene_name(scene), f"""
                <header><a href="/">All models</a><b>/{number} · {html.escape(scene_name(scene))}</b>
                {previous}{following}<select id="mode" aria-label="Model view">{options}</select>
                <a href="/{number}/multiview/scene_colored.glb" download>Download GLB</a></header>
                <iframe id="model" title="Interactive 3D reconstruction" src="/{number}/multiview/mesh_viewer.html"></iframe>
                <script>document.getElementById('mode').onchange=e=>{{document.getElementById('model').src=e.target.value}};</script>
                """, "viewer")
        target = (scene / relative).resolve()
        # Do not expose other runs, parent directories, or symlink targets outside this reconstruction.
        if not target.is_relative_to(scene) or not target.is_file():
            self.send_error(404, "File not found")
            return None
        if target.name in {"mesh_viewer.html", "supported_viewer.html", "textured_viewer.html"}:
            return self.respond(compact_viewer(target.read_text()).encode(), "text/html; charset=utf-8")
        self.file_target = target
        return super().send_head()

    def translate_path(self, path):
        return str(self.file_target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    scenes = [p.resolve() for p in sorted(BATCH.glob("*/reconstruction"))
              if p.is_dir() and (p / "multiview/mesh_viewer.html").is_file()
              and p.resolve().is_relative_to(BATCH.resolve())]
    if (EXTRA / "multiview/mesh_viewer.html").is_file():
        scenes.append(EXTRA.resolve())
    if not scenes:
        parser.error(f"No model viewers found in {BATCH}")
    server = ThreadingHTTPServer((args.host, args.port), partial(BatchHandler, scenes=scenes))
    print(f"Serving {len(scenes)} reconstructions on http://{args.host}:{args.port}", flush=True)
    for i, scene in enumerate(scenes, 1):
        print(f"/{i}: {scene_name(scene)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
