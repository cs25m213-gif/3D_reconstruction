"""Create a local, browser-based viewer for an OpenMVS PLY mesh."""
import argparse
import json
import math
import re
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED


def write_mesh_viewer(mesh_dir: Path, three_assets: Path):
    mesh_dir, three_assets = Path(mesh_dir).resolve(), Path(three_assets).resolve()
    if not (mesh_dir / 'scene_mesh.ply').is_file():
        raise FileNotFoundError(mesh_dir / 'scene_mesh.ply')
    for name in ('three.module.js', 'PLYLoader.js', 'OrbitControls.js',
                 'BufferGeometryUtils.js'):
        shutil.copy2(three_assets / name, mesh_dir / name)
    shutil.copy2(three_assets / 'GLTFLoader_local.js', mesh_dir / 'GLTFLoader.js')
    metrics = json.loads((mesh_dir / 'dense_metrics.json').read_text())
    observation_path=mesh_dir/'object_observations.json'
    observations=json.loads(observation_path.read_text()) if observation_path.is_file() else None
    with (mesh_dir / 'scene_mesh.ply').open('rb') as stream:
        header = stream.read(4096).split(b'end_header', 1)[0].decode('ascii', 'replace')
    for element, key in (('face', 'mesh_faces'), ('vertex', 'mesh_vertices')):
        match = re.search(r'^element ' + element + r' (\d+)$', header, re.M)
        if match:
            metrics[key] = int(match.group(1))
    (mesh_dir / 'dense_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    textured = (mesh_dir / 'scene_textured_glb.glb').is_file()
    quality_link = ('<a href="quality/index.html">Coverage report</a>'
                    if (mesh_dir / 'quality/index.html').is_file() else '')
    if (mesh_dir / 'hybrid_report.json').is_file():
        quality_link += '<a href="hybrid_report.json">Hybrid report</a>'
    if (mesh_dir / 'hybrid_dense_report.json').is_file():
        quality_link += '<a href="hybrid_dense_report.json">Hybrid coverage report</a>'
    inferred = (mesh_dir / 'scene_inferred_planes.glb').is_file()
    hybrid_layers = [[ident, name, label, is_ply] for ident, name, label, is_ply in (
        ('hybrid-mesh', 'scene_hybrid_mesh.glb', 'hybrid mesh (measured + prior)', False),
        ('hybrid-provenance', 'scene_hybrid_provenance.ply', 'provenance colours', True),
        ('hybrid-planes', 'scene_hybrid_planes.glb', 'hybrid planes (inferred)', False))
        if (mesh_dir / name).is_file()]
    for ident, _, label, _ in hybrid_layers:
        quality_link += f'<button id="{ident}">Show {label}</button>'
    colored_ply = (mesh_dir / 'scene_mesh_colored.ply').is_file()
    ply_name = 'scene_mesh_colored.ply' if colored_ply else 'scene_mesh.ply'
    if (mesh_dir / 'scene_colored.glb').is_file():
        quality_link += '<a href="scene_colored.glb" download>Download colored GLB</a>'
    if observations is not None:
        quality_link += '<button id="objects-toggle">Hide object observations</button><a href="object_observations.json">Object observations</a><a href="object_markers.ply" download>Download markers</a>'
    preview = None
    model_path = Path(metrics['source_model']) if metrics.get('source_model') else None
    if model_path is not None and model_path.is_dir():
        import pycolmap
        reconstruction = pycolmap.Reconstruction(model_path)
        registered = [image for image in reconstruction.images.values() if image.has_pose]
        if registered:
            image = max(registered, key=lambda item: item.num_points3D)
            pose = image.cam_from_world().inverse()
            rotation = pose.rotation.matrix()
            camera_model = reconstruction.cameras[image.camera_id]
            fy = camera_model.calibration_matrix()[1, 1]
            preview = {'frame': image.name, 'position': pose.translation.tolist(),
                       'forward': rotation[:, 2].tolist(), 'up': (-rotation[:, 1]).tolist(),
                       'fov_y': math.degrees(2 * math.atan(camera_model.height / (2 * fy)))}
    page = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GeoStreamFusion dense mesh</title>
<style>*{box-sizing:border-box}body{margin:0;background:#0e151c;color:#e2ebf3;font:14px system-ui}
header{padding:12px 18px;background:#17222c;border-bottom:1px solid #304252;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
b{font-size:17px}a{color:#83d8ff}button{background:#345066;color:white;border:0;border-radius:4px;padding:6px 9px;cursor:pointer}#status{color:#a6bed1}#view{height:calc(100vh - 62px)}
canvas{display:block}small{color:#a6bed1}</style>
<header><b>GeoStreamFusion dense mesh</b><span id="summary"></span><span id="status">Loading mesh…</span>
<button id="capture-view">Capture viewpoint</button><button id="overview">Overview</button>VIEW_LINKS<a href="PLY_NAME" download>Download colored PLY</a><a href="dense_mesh_bundle.zip" download>Download mesh bundle</a><small>Drag to rotate · scroll to zoom · right drag to pan</small></header>
<div id="view"></div>
<script type="importmap">{"imports":{"three":"./three.module.js"}}</script>
<script type="module">
import * as THREE from 'three';
import {PLYLoader} from './PLYLoader.js';
import {GLTFLoader} from './GLTFLoader.js';
import {OrbitControls} from './OrbitControls.js';
const metrics = METRICS;
const hasTexture = HAS_TEXTURE;
const hasInferred = HAS_INFERRED;
const preview = PREVIEW;
const observationReport = OBJECT_OBSERVATIONS;
const objectRows = (observationReport?.observations || []).filter(r=>Array.isArray(r.position)&&r.position.length===3&&r.position.every(Number.isFinite));
if(observationReport){const note=document.createElement('small');note.textContent=`${objectRows.length} estimated object positions across frames; ${observationReport.unlocalized_observations} detections without usable depth. Orange: person; green: animal. Hover markers for frame/time. These are observations, not unique individuals.`;document.querySelector('header').appendChild(note)}
document.getElementById('summary').textContent = ` ${metrics.mesh_faces.toLocaleString()} faces` + (metrics.dense_points ? ` · ${metrics.dense_points.toLocaleString()} fused points` : ' · hybrid depth');
if(metrics.coloring){document.getElementById('summary').textContent = `${metrics.mesh_faces.toLocaleString()} faces · RGB color on ${Math.round(100*metrics.coloring.colored_vertex_fraction)}% of vertices`}
if(metrics.quality_warning){const warning=document.createElement('small');warning.textContent=metrics.quality_warning;document.querySelector('header').appendChild(warning)}
const container=document.getElementById('view');
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0e151c);
const camera=new THREE.PerspectiveCamera(55,1,.001,1e7);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));
container.appendChild(renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff,2));
for(const [x,y,z] of [[1,1,1],[-1,.5,-1]]){const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(x,y,z);scene.add(light)}
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
function resize(){renderer.setSize(container.clientWidth,container.clientHeight);camera.aspect=container.clientWidth/container.clientHeight;camera.updateProjectionMatrix()}
addEventListener('resize',resize);resize();
function display(object){
 const box=new THREE.Box3().setFromObject(object),center=box.getCenter(new THREE.Vector3());
 object.position.sub(center);
 const radius=Math.max(box.getSize(new THREE.Vector3()).length()*.65,.001);
 camera.near=Math.max(radius/10000,.00001);camera.far=radius*100;
 function overview(){camera.up.set(0,1,0);camera.fov=55;camera.position.set(radius,radius*.7,radius);controls.target.set(0,0,0);camera.updateProjectionMatrix();controls.update()}
 function captureView(){if(!preview){overview();return}
  camera.up.fromArray(preview.up);camera.fov=preview.fov_y;
  camera.position.fromArray(preview.position).sub(center);
  controls.target.copy(camera.position).addScaledVector(new THREE.Vector3().fromArray(preview.forward),radius*.5);
  camera.updateProjectionMatrix();controls.update()}
 document.getElementById('overview').onclick=overview;
 document.getElementById('capture-view').onclick=captureView;
 preview?captureView():overview();
 scene.add(object);document.getElementById('status').textContent='Mesh loaded';
 if(observationReport){
  const positions=[],colors=[];
  for(const row of objectRows){positions.push(...new THREE.Vector3().fromArray(row.position).sub(center).toArray());colors.push(...new THREE.Color(row.class==='person'?0xff8c1e:0x5af078).toArray())}
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
  const markers=new THREE.Points(geometry,new THREE.PointsMaterial({size:radius*.008,vertexColors:true,depthTest:true}));scene.add(markers);
  const button=document.getElementById('objects-toggle');button.onclick=()=>{markers.visible=!markers.visible;button.textContent=markers.visible?'Hide object observations':'Show object observations'};
  const picker=new THREE.Raycaster();picker.params.Points.threshold=radius*.008;
  renderer.domElement.addEventListener('pointermove',event=>{
   const rect=renderer.domElement.getBoundingClientRect();picker.setFromCamera(new THREE.Vector2(2*(event.clientX-rect.left)/rect.width-1,1-2*(event.clientY-rect.top)/rect.height),camera);
   const hit=markers.visible?picker.intersectObject(markers)[0]:null;
   if(!hit){document.getElementById('status').textContent='Mesh loaded';return}
   const row=objectRows[hit.index],time=row.timestamp_seconds==null?'time unknown':`${row.timestamp_seconds.toFixed(2)} s`;
   document.getElementById('status').textContent=`${row.class} · ${Math.round(100*row.confidence)}% detection confidence · ${time} · ${row.frame} · ${row.position_status.replaceAll('_',' ')}`;
  });
 }
 if(hasInferred){let inferred=null;const button=document.getElementById('inferred-toggle');
  button.onclick=()=>{if(inferred){inferred.visible=!inferred.visible;
   button.textContent=inferred.visible?'Hide estimated planes':'Show estimated planes';return}
   button.disabled=true;new GLTFLoader().load('./scene_inferred_planes.glb',data=>{
    inferred=data.scene;inferred.position.sub(center);inferred.traverse(child=>{
     if(child.isMesh){child.material.side=THREE.DoubleSide;child.material.transparent=true;
      child.material.opacity=.65;child.material.depthWrite=false}});
    scene.add(inferred);button.disabled=false;button.textContent='Hide estimated planes';
   },progress,error=>{button.disabled=false;failure(error)})}}
 for(const [id,file,label,isPly] of HYBRID_LAYERS){const b=document.getElementById(id);if(!b)continue;let layer=null;
  b.onclick=()=>{if(layer){layer.visible=!layer.visible;b.textContent=(layer.visible?'Hide ':'Show ')+label;return}
   b.disabled=true;const done=o=>{layer=o;o.position.sub(center);scene.add(o);b.disabled=false;b.textContent='Hide '+label};
   const fail=e=>{b.disabled=false;failure(e)};
   if(isPly){new PLYLoader().load('./'+file,g=>{g.computeVertexNormals();done(new THREE.Mesh(g,new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.DoubleSide})))},progress,fail)}
   else{new GLTFLoader().load('./'+file,d=>{d.scene.traverse(c=>{if(c.isMesh)c.material.side=THREE.DoubleSide});done(d.scene)},progress,fail)}}}
}
function progress(event){if(event.total)document.getElementById('status').textContent=`Loading ${Math.round(100*event.loaded/event.total)}%`}
function failure(error){document.getElementById('status').textContent=`Could not load mesh: ${error.message}`}
if(hasTexture){new GLTFLoader().load('./scene_textured_glb.glb',data=>{
 data.scene.traverse(child=>{if(child.isMesh)child.material.side=THREE.DoubleSide});display(data.scene)
},progress,failure)}
else{new PLYLoader().load('./PLY_NAME',geometry=>{
 if(!geometry.hasAttribute('normal'))geometry.computeVertexNormals();
 display(new THREE.Mesh(geometry,geometry.hasAttribute('color') ? new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.DoubleSide}) : new THREE.MeshStandardMaterial({color:0xd8dfe4,side:THREE.DoubleSide,roughness:1})))
},progress,failure)}
function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera)}animate();
</script></html>'''.replace('METRICS', json.dumps(metrics).replace('<', '\\u003c')).replace('PREVIEW', json.dumps(preview)).replace('PLY_NAME', ply_name)
    path = mesh_dir / 'mesh_viewer.html'
    page = page.replace('HAS_INFERRED', 'true' if inferred else 'false').replace(
        'HYBRID_LAYERS', json.dumps(hybrid_layers))
    page = page.replace('OBJECT_OBSERVATIONS',json.dumps(observations,allow_nan=False).replace('<','\\u003c'))
    if inferred:
        quality_link += '<button id="inferred-toggle" title="Plane estimates are separate from measured geometry">Show estimated planes</button>'
    if colored_ply:
        path.write_text(page.replace('HAS_TEXTURE', 'false').replace(
            'VIEW_LINKS', quality_link + ('<a href="textured_viewer.html">Texture view</a>' if textured else '')))
        if textured:
            (mesh_dir / 'textured_viewer.html').write_text(page.replace(
                'HAS_TEXTURE', 'true').replace('VIEW_LINKS', quality_link + '<a href="mesh_viewer.html">Vertex color view</a>'))
    else:
        path.write_text(page.replace('HAS_TEXTURE', 'true' if textured else 'false').replace('VIEW_LINKS', quality_link))
    names = ['scene_mesh.ply', 'dense_metrics.json', 'mesh_viewer.html',
             'three.module.js', 'PLYLoader.js', 'OrbitControls.js',
             'BufferGeometryUtils.js', 'GLTFLoader.js']
    if observations is not None:
        names += ['object_observations.json','object_markers.ply']
    if (mesh_dir / 'scene_colored.glb').exists():
        names.append('scene_colored.glb')
    if inferred:
        names.extend(['scene_inferred_planes.glb', 'planar_completion_report.json'])
    if colored_ply:
        names.append('scene_mesh_colored.ply')
    if textured:
        names += ['scene_textured_glb.glb']
        names += [p.name for p in sorted(mesh_dir.glob('*.png'))]
        if colored_ply:
            names.append('textured_viewer.html')
    if (mesh_dir / 'quality_report.json').exists():
        names.append('quality_report.json')
        names.extend(str(p.relative_to(mesh_dir)) for p in sorted((mesh_dir / 'quality').glob('*')) if p.is_file())
    if (mesh_dir / 'hybrid_report.json').exists():
        names.append('hybrid_report.json')
    names.extend(name for _, name, _, _ in hybrid_layers)
    if (mesh_dir / 'hybrid_dense_report.json').exists():
        names.append('hybrid_dense_report.json')
        names.extend(str(p.relative_to(mesh_dir)) for p in sorted((mesh_dir / 'quality').glob('*'))
                     if p.is_file() and (p.name == 'index.html' or p.name.startswith('hybrid_')))
    with ZipFile(mesh_dir / 'dense_mesh_bundle.zip', 'w', ZIP_DEFLATED, compresslevel=6) as archive:
        for name in dict.fromkeys(names):
            archive.write(mesh_dir / name, arcname=name)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh_dir', type=Path)
    parser.add_argument('--three-assets', type=Path, required=True)
    args = parser.parse_args()
    print(write_mesh_viewer(args.mesh_dir, args.three_assets))


if __name__ == '__main__':
    main()
