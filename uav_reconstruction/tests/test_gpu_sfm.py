import unittest
from types import SimpleNamespace
from unittest.mock import patch
import pycolmap
from adaptive_uav.gpu_sfm import configure_mapping_ba


class GpuSfMTests(unittest.TestCase):
    def test_unverified_build_stays_cpu_for_bundle_adjustment(self):
        with patch('adaptive_uav.gpu_sfm.verified_capabilities',return_value={}):
            options=pycolmap.IncrementalPipelineOptions()
            result=configure_mapping_ba(options,{'sfm_use_gpu':True},pycolmap)
            self.assertFalse(result['gpu_bundle_adjustment'])
            self.assertEqual(options.ba_global_backend,pycolmap.BundleAdjustmentBackend.CERES)

    def test_explicit_gpu_solver_cannot_silently_fall_back(self):
        with patch('adaptive_uav.gpu_sfm.verified_capabilities',return_value={}):
            with self.assertRaises(RuntimeError):
                configure_mapping_ba(pycolmap.IncrementalPipelineOptions(),
                    {'sfm_use_gpu':True,'sfm_ba_backend':'caspar'},pycolmap)

    def test_verified_caspar_selects_gpu_for_local_and_global_ba(self):
        with patch('adaptive_uav.gpu_sfm.verified_capabilities',return_value={'caspar_verified':True}):
            options=pycolmap.IncrementalPipelineOptions()
            result=configure_mapping_ba(options,{'sfm_use_gpu':True,'sfm_ba_backend':'caspar'},pycolmap,
                                        camera_models=['PINHOLE'])
            self.assertEqual(options.ba_local_backend,pycolmap.BundleAdjustmentBackend.CASPAR)
            self.assertEqual(options.ba_global_backend,pycolmap.BundleAdjustmentBackend.CASPAR)
            self.assertTrue(result['gpu_bundle_adjustment'])
            self.assertEqual(result['camera_registration_device'],'CPU')

    def test_unsupported_caspar_model_uses_ceres_instead_of_skipping_cameras(self):
        with patch('adaptive_uav.gpu_sfm.verified_capabilities',return_value={'caspar_verified':True,'ceres_gpu_verified':True}):
            options=pycolmap.IncrementalPipelineOptions()
            result=configure_mapping_ba(options,{'sfm_use_gpu':True,'sfm_ba_backend':'caspar'},pycolmap,
                                        camera_models=['SIMPLE_PINHOLE'])
            self.assertEqual(options.ba_global_backend,pycolmap.BundleAdjustmentBackend.CERES)
            self.assertTrue(result['gpu_bundle_adjustment'])
            self.assertIn('SIMPLE_PINHOLE',result['fallback_reason'])

    def test_caspar_cannot_run_without_camera_compatibility_information(self):
        with patch('adaptive_uav.gpu_sfm.verified_capabilities',return_value={'caspar_verified':True}):
            with self.assertRaises(ValueError):
                configure_mapping_ba(pycolmap.IncrementalPipelineOptions(),
                    {'sfm_use_gpu':True,'sfm_ba_backend':'caspar'},pycolmap)

    def test_auto_uses_verified_caspar_for_supported_camera(self):
        with patch('adaptive_uav.gpu_sfm.verified_capabilities',return_value={'caspar_verified':True,'ceres_gpu_verified':True}):
            options=pycolmap.IncrementalPipelineOptions()
            result=configure_mapping_ba(options,{'sfm_use_gpu':True,'prefer_caspar_auto':True},pycolmap,
                                        camera_models=['PINHOLE'])
            self.assertTrue(result['gpu_bundle_adjustment'])
            self.assertEqual(options.ba_global_backend,pycolmap.BundleAdjustmentBackend.CASPAR)

    def test_auto_falls_back_for_unsupported_camera(self):
        with patch('adaptive_uav.gpu_sfm.verified_capabilities',return_value={'caspar_verified':True,'ceres_gpu_verified':True}):
            options=pycolmap.IncrementalPipelineOptions()
            result=configure_mapping_ba(options,{'sfm_use_gpu':True,'prefer_caspar_auto':True},pycolmap,
                                        camera_models=['SIMPLE_PINHOLE'])
            self.assertEqual(options.ba_global_backend,pycolmap.BundleAdjustmentBackend.CERES)
            self.assertIn('SIMPLE_PINHOLE',result['fallback_reason'])


if __name__=='__main__':unittest.main()
