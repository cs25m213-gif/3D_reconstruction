"""Native module ownership must survive both import orders and package aliases."""
import os,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

class Open3DImportOrderTests(unittest.TestCase):
    def run_child(self,source):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        result=subprocess.run([sys.executable,'-c',source],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_normal_import_then_cpu_helper(self):
        self.run_child('import open3d; from adaptive_uav.open3d_cpu import cpu_bindings; assert cpu_bindings().geometry.TriangleMesh is open3d.geometry.TriangleMesh')

    def test_cpu_helper_then_normal_import(self):
        self.run_child("import sys; from adaptive_uav.open3d_cpu import cpu_bindings; cpu=cpu_bindings(); assert 'open3d' not in sys.modules; import open3d; assert cpu.geometry.TriangleMesh is open3d.geometry.TriangleMesh")

    def test_two_project_import_names_share_extension(self):
        self.run_child('from adaptive_uav.open3d_cpu import cpu_bindings as a; from uav_reconstruction.adaptive_uav.open3d_cpu import cpu_bindings as b; assert a() is b()')

if __name__=='__main__':unittest.main()
