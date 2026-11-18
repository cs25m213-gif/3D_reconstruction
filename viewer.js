import * as THREE from './assets/three.module.js';
import {GLTFLoader} from './assets/GLTFLoader.js';
import {TrackballControls} from './assets/TrackballControls.js';
let resumeViewer=null;
export async function mountViewer(){
 if(resumeViewer)return resumeViewer();
 const status=document.querySelector('#viewer-status'),host=document.querySelector('#canvas-host'),box=document.querySelector('#viewer');
 const base=new URL('./assets/campus-stream/',import.meta.url);
 const manifest=await fetch(new URL('manifest.json?v=4',base)).then(r=>{if(!r.ok)throw new Error('Model manifest unavailable.');return r.json()});
 const preview=await loadTexture(new URL(manifest.previewTexture.url,base));
 const material=new THREE.MeshBasicMaterial({map:preview,side:THREE.DoubleSide});
 const model=new THREE.Group();
 const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.outputColorSpace=THREE.SRGBColorSpace;host.replaceChildren(renderer.domElement);
 const scene=new THREE.Scene();scene.background=new THREE.Color(0x000000);const camera=new THREE.PerspectiveCamera(52.974,1,.001,10000);
 const bounds=new THREE.Box3(new THREE.Vector3(...manifest.bounds.min),new THREE.Vector3(...manifest.bounds.max)),center=bounds.getCenter(new THREE.Vector3()),radius=bounds.getSize(new THREE.Vector3()).length()/2;model.position.sub(center);scene.add(model);scene.add(new THREE.HemisphereLight(0xffffff,0x777777,2));
 const control=new TrackballControls(camera,renderer.domElement);control.rotateSpeed=3.5;control.zoomSpeed=1.25;control.panSpeed=.6;control.staticMoving=true;
 function reset(){camera.position.set(4.003660790560263,.45714539015284333,-1.0447497568223296).sub(center);camera.up.set(-.1942869002497792,-.9801955786999433,-.038330508585401574);control.target.copy(camera.position).addScaledVector(new THREE.Vector3(-.8608483492738983,.1516358237290149,.4857434472171781),radius*.5);control.target.addScaledVector(camera.up,-radius*.065);camera.zoom=1.08;camera.near=Math.max(radius/10000,.00001);camera.far=radius*200;camera.updateProjectionMatrix();control.update()}
 function resize(){const w=box.clientWidth,h=box.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();control.handleResize()}
 new ResizeObserver(resize).observe(host);resize();reset();

 document.querySelector('#reset-view').onclick=reset;let wire=false;document.querySelector('#wire-view').onclick=()=>{wire=!wire;model.traverse(o=>{if(o.isMesh){const mats=Array.isArray(o.material)?o.material:[o.material];for(const m of mats)m.wireframe=wire}});document.querySelector('#wire-view').setAttribute('aria-pressed',wire)};
 document.querySelector('#full-view').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await box.requestFullscreen()}catch{status.textContent='Fullscreen is unavailable in this browser.'}};
 let visible=true;new IntersectionObserver(([e])=>{visible=e.isIntersecting}).observe(box);function render(){requestAnimationFrame(render);if(visible){control.update();renderer.render(scene,camera)}}render();
 const complete=new Set();let textureReady=false,shown=false;
 const loader=new GLTFLoader();
 const byteProgress=new Map();
 const button=document.querySelector('#load-model');
 function progress(){
  const received=[...byteProgress.values()].reduce((a,b)=>a+b,0);
  status.textContent=`${complete.size}/${manifest.parts.length} model pieces ready · ${(received/1e6).toFixed(1)} MB received${shown?' · you can rotate now':''}`;
  button.textContent=`Loading ${Math.round(100*received/(manifest.totalBytes-manifest.previewTexture.bytes))}%`;
 }
 async function run(){
  const parts=manifest.parts.filter(p=>!complete.has(p.url));
  const jobs=[...parts.slice(0,4),...(!textureReady?[{...manifest.texture,texture:true}]:[]),...parts.slice(4)];
  let next=0;const failures=[];
  async function worker(){
   while(next<jobs.length){const job=jobs[next++];
    try{
     const bytes=await download(new URL(job.url,base),n=>{byteProgress.set(job.url,n);progress()});
     if(bytes.byteLength!==job.bytes)throw new Error('Incomplete model download.');
     if(job.texture){const tex=await textureFromBytes(bytes);const old=material.map;material.map=tex;material.needsUpdate=true;old.dispose();old.image.close?.();textureReady=true;}
     else{
      const buffer=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
      const gltf=await loader.parseAsync(buffer,'');
      gltf.scene.traverse(o=>{if(o.isMesh){o.material.dispose();o.material=material;}});
      material.wireframe=wire;model.add(gltf.scene);complete.add(job.url);
      if(!shown){shown=true;host.hidden=false;resize();document.querySelector('#viewer-preview').hidden=true;document.querySelector('#viewer-overlay').hidden=true;document.querySelector('#viewer-overlay').style.display='none';document.querySelector('#viewer-tools').hidden=false;window.geostreamfusionFirstPieceMs=performance.now();}
      await new Promise(requestAnimationFrame);
     }
     progress();
    }catch(e){failures.push(e);}
   }
  }
  await Promise.all(Array.from({length:Math.min(4,jobs.length)},worker));
  if(failures.length){
   status.textContent=`${complete.size}/${manifest.parts.length} pieces loaded. Connection interrupted; retry the missing pieces.`;
   document.querySelector('#viewer-overlay').hidden=false;document.querySelector('#viewer-overlay').style.display='';
   throw failures[0];
  }
  window.geostreamfusionModelReady=true;window.geostreamfusionModelReadyMs=performance.now();
  status.textContent='Model ready. Left drag: free rotation · right drag: pan · scroll/pinch: zoom.';
  document.querySelector('#viewer-overlay').hidden=true;document.querySelector('#viewer-overlay').style.display='none';
 }
 resumeViewer=run;
 await run();
}

async function download(url,onProgress=()=>{}){
 for(let attempt=0;attempt<3;attempt++){
  const controller=new AbortController();const timeout=setTimeout(()=>controller.abort(),45000);
  try{
   onProgress(0);const r=await fetch(url,{signal:controller.signal});if(!r.ok)throw new Error(`Download failed (${r.status}).`);
   const reader=r.body.getReader(),parts=[];let n=0;
   for(;;){const {done,value}=await reader.read();if(done)break;parts.push(value);n+=value.byteLength;onProgress(n);}
   return await new Blob(parts).arrayBuffer();
  }catch(e){if(attempt===2)throw e;await new Promise(r=>setTimeout(r,400*(attempt+1)));}
  finally{clearTimeout(timeout);}
 }
}
async function textureFromBytes(bytes){
 const bitmap=await createImageBitmap(new Blob([bytes],{type:'image/jpeg'}));
 const texture=new THREE.Texture(bitmap);texture.colorSpace=THREE.SRGBColorSpace;texture.flipY=false;
 texture.magFilter=THREE.LinearFilter;texture.minFilter=THREE.LinearFilter;texture.generateMipmaps=false;texture.needsUpdate=true;return texture;
}
async function loadTexture(url){return textureFromBytes(await download(url));}
