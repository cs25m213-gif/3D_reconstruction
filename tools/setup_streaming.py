#!/usr/bin/env python3
"""Install the optional pinned DPVO worker without changing the main environment.

Run with the system Python from the repository root. Shares existing Torch and
depth dependencies via .pth; new native dependencies live in .venv_dpvo only.
"""
import argparse,hashlib,io,json,os,re,subprocess,tarfile,urllib.request,venv,zipfile
from pathlib import Path

COMMIT='0ac95b656d1fda91c271d2a106460d19ad966fc7'
CHECKPOINT_SHA='30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12'
MODEL_URL='https://www.dropbox.com/scl/fo/2g6e134ruh1acmaye5bmy/AK0DY9WLZ25elq-7N7sJcB0/models.zip?rlkey=4j336vcc4h324j4p4ji3ppl0t&dl=1'


def download(url):
    request=urllib.request.Request(url,headers={'User-Agent':'GeoStreamFusion optional backend installer'})
    with urllib.request.urlopen(request,timeout=120) as response:return response.read()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];repo=root/'tools/DPVO';env=root/'.venv_dpvo'
    def run(command,**kwargs):subprocess.run(list(map(str,command)),cwd=root,check=True,**kwargs)
    if not args.verify_only:
        if not repo.exists():run(['git','clone','https://github.com/princeton-vl/DPVO.git',repo])
        actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if actual!=COMMIT:
            if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():
                raise RuntimeError('DPVO checkout has edits at another revision; preserve it and select the pinned revision deliberately')
            run(['git','-C',repo,'checkout',COMMIT])
        if not env.exists():venv.EnvBuilder(with_pip=True,system_site_packages=True).create(env)
        shared=next((root/'.venv/lib').glob('python*/site-packages'));site=next((env/'lib').glob('python*/site-packages'))
        (site/'geostream_shared.pth').write_text(str(shared)+'\n')
        third=repo/'thirdparty';third.mkdir(exist_ok=True);eigen=third/'eigen-3.4.0'
        if not eigen.exists():
            with tarfile.open(fileobj=io.BytesIO(download('https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz')),mode='r:gz') as archive:
                archive.extractall(third,filter='data')
        patches=[]
        for path in (repo/'dpvo').rglob('*'):
            if path.suffix not in ('.cu','.cpp','.h'):continue
            text=path.read_text();new=re.sub(r'(\b\w+)\.type\(\)(?=,)',r'\1.scalar_type()',text)
            if text!=new:path.write_text(new)
            if '.scalar_type()' in new:patches.append(str(path.relative_to(repo)))
        checkpoint=repo/'dpvo.pth'
        if not checkpoint.exists():
            data=download(MODEL_URL)
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                name=next(n for n in archive.namelist() if n.endswith('dpvo.pth'));checkpoint.write_bytes(archive.read(name))
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA:raise ValueError('Unexpected DPVO checkpoint hash')
        run([env/'bin/pip','install','yacs==0.1.8','ninja==1.13.2','numba==0.68.0','pypose==0.9.5','kornia==0.9.0'])
        # Wheel selection follows the installed Torch/CUDA pair. Build-from-source
        # attempts must not silently upgrade Torch in the shared environment.
        details=json.loads(subprocess.check_output([str(env/'bin/python'),'-c','import torch,json;print(json.dumps([torch.__version__.split("+")[0],torch.version.cuda]))'],text=True))
        torch_version,cuda=details;tag='cu'+cuda.replace('.','') if cuda else 'cpu'
        run([env/'bin/pip','install','--no-deps','torch-scatter','-f',f'https://data.pyg.org/whl/torch-{torch_version}+{tag}.html'])
        build_env=dict(os.environ,MAX_JOBS='2')
        if 'TORCH_CUDA_ARCH_LIST' not in build_env:
            capability=subprocess.check_output([str(env/'bin/python'),'-c','import torch;print(".".join(map(str,torch.cuda.get_device_capability())))'],text=True).strip()
            build_env['TORCH_CUDA_ARCH_LIST']=capability
        run([env/'bin/pip','install','--no-build-isolation','--no-deps',repo],env=build_env)
        (repo/'geostream_provenance.json').write_text(json.dumps({'repository':'https://github.com/princeton-vl/DPVO','commit':COMMIT,
            'license':'MIT repository; checkpoint from authors published model archive','checkpoint_sha256':CHECKPOINT_SHA,
            'torch_compatibility_patches':patches,'patch_file_sha256':{p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in patches}},indent=2))
    verify_env=dict(os.environ,PYTHONPATH=str(repo))
    run([env/'bin/python','-c','from dpvo.dpvo import DPVO;from dpvo.config import cfg;import cuda_corr,cuda_ba,lietorch_backends;print("DPVO CUDA imports verified")'],env=verify_env)
    print('Optional DPVO environment ready. Actual video/geometry validation remains separate.')


if __name__=='__main__':main()
