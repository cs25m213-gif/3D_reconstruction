"""Optional local streaming viewer. Poll metadata and transfer changed chunks only."""
from pathlib import Path
import shutil


def write_stream_viewer(output,assets):
    output=Path(output);assets=Path(assets)
    for name in ('three.module.js','PLYLoader.js','OrbitControls.js','BufferGeometryUtils.js'):
        shutil.copy2(assets/name,output/name)
    (output/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>GeoStreamFusion live map</title>
<style>body{margin:0;background:#101720;color:#eee;font:15px system-ui}header{padding:12px}a{color:#8df}#view{height:88vh}canvas{display:block}</style>
<header><strong>GeoStreamFusion incremental map</strong> <span id="status">Waiting for map…</span>
<a href="multiview/mesh_viewer.html">Final mesh + object observations</a> · <a href="run_report.json">Run report</a>
<div>Estimated geometry in arbitrary visual units. Drag to orbit; scroll to zoom. Relative pose corrections can require rebuilding chunks.</div>
<div id="metrics"></div><img id="rgb" style="max-width:24vw" alt="Latest tracked frame"><img id="depth" style="max-width:24vw" alt="Latest estimated depth; relative visual units"></header><div id="view"></div>
<script type="importmap">{"imports":{"three":"./three.module.js"}}</script><script type="module">
import * as THREE from 'three';import {PLYLoader} from './PLYLoader.js';import {OrbitControls} from './OrbitControls.js';
const view=document.getElementById('view'),scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(55,1,.001,1e6);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));view.appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;const layers=new Map(),loader=new PLYLoader();let fitted=false;
const trailGeometry=new THREE.BufferGeometry(),trail=new THREE.Line(trailGeometry,new THREE.LineBasicMaterial({color:0x66ff99}));scene.add(trail);
function resize(){renderer.setSize(view.clientWidth,view.clientHeight);camera.aspect=view.clientWidth/view.clientHeight;camera.updateProjectionMatrix()}addEventListener('resize',resize);resize();
async function poll(){try{const response=await fetch('./map_updates.json',{cache:'no-store'});if(!response.ok)return;const update=await response.json();
document.getElementById('status').textContent=`${update.status} · update ${update.sequence} · ${update.chunks.length} chunks`;
document.getElementById('metrics').textContent=`Tracking: ${update.tracking_status||'unknown'} · ${update.selected_keyframes||0} mapping keyframes · ${(update.mapping_updates_per_second||0).toFixed(1)} map updates/s · queued ${update.queue_length||0} · peak process-tree RAM ${((update.resources?.peak_tree_rss_bytes||0)/1073741824).toFixed(2)} GB`;
document.getElementById('rgb').src='./current_rgb.jpg?v='+update.sequence;document.getElementById('depth').src='./current_depth.jpg?v='+update.sequence;
if(update.trajectory?.length){trailGeometry.setAttribute('position',new THREE.Float32BufferAttribute(update.trajectory.flat(),3));trailGeometry.computeBoundingSphere()}
const ids=new Set(update.chunks.map(c=>c.id));for(const [id,old] of layers){if(!ids.has(id)){scene.remove(old.mesh);old.mesh.geometry.dispose();old.mesh.material.dispose();layers.delete(id)}}
await Promise.all(update.chunks.map(chunk=>new Promise(resolve=>{const old=layers.get(chunk.id);if(old?.version===chunk.version){resolve();return}
loader.load('./'+chunk.url+'?v='+chunk.version,geometry=>{if(old){scene.remove(old.mesh);old.mesh.geometry.dispose();old.mesh.material.dispose()}
const mesh=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.DoubleSide,transparent:chunk.stale,opacity:chunk.stale?.4:1}));
scene.add(mesh);layers.set(chunk.id,{mesh,version:chunk.version});if(!fitted){const box=new THREE.Box3().setFromObject(mesh),center=box.getCenter(new THREE.Vector3()),size=Math.max(box.getSize(new THREE.Vector3()).length(),.01);camera.position.copy(center).add(new THREE.Vector3(size,size,size));controls.target.copy(center);camera.updateProjectionMatrix();fitted=true}resolve()},undefined,()=>resolve())})));}catch(error){document.getElementById('status').textContent=error.message}}
setInterval(poll,1000);poll();function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera)}animate();
</script>''')
