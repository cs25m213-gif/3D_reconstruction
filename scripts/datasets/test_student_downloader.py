"""Small storage/protection checks; no real dataset downloads or GPU use."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import download_student_datasets as downloader


def archive_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state = self.root / 'state'; self.state.mkdir()
        self.extracted = self.root / 'extracted'; self.extracted.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def asset(self, data, kind='calibration'):
        return {'name': 'tiny.zip', 'kind': kind, 'url': 'https://example.invalid/tiny.zip', 'bytes': len(data)}

    def test_default_scope_excludes_global_depth_and_is_bounded(self):
        manifest = json.loads((Path(__file__).parent / 'student_datasets_manifest.json').read_text())
        assets = downloader.plan(manifest)
        self.assertEqual(len(assets), 5)
        self.assertEqual(sum(a['bytes'] for a in assets), 2238819967)
        self.assertTrue(all(not a['optional_depth'] for a in assets))
        self.assertEqual(len(downloader.plan(manifest, calibration_only=True)), 2)
        self.assertEqual(len(downloader.plan(manifest, with_depth=True)), 6)

    def test_real_zip_crc_extraction_and_reuse(self):
        data = archive_bytes({'date/calib.txt': b'calibration'})
        path = self.root / 'tiny.zip'; path.write_bytes(data)
        result = downloader.verify_extract(self.asset(data), path, self.extracted, self.state, [])
        self.assertEqual((self.extracted / 'date/calib.txt').read_bytes(), b'calibration')
        self.assertEqual(result['full_zip_crc'], 'pass')
        with mock.patch.object(zipfile.ZipFile, 'testzip', side_effect=AssertionError('should reuse verified archive')):
            again = downloader.verify_extract(self.asset(data), path, self.extracted, self.state, [])
        self.assertEqual(result, again)

    def test_changed_extracted_file_is_not_overwritten(self):
        data = archive_bytes({'calib.txt': b'correct'})
        path = self.root / 'tiny.zip'; path.write_bytes(data)
        target = self.extracted / 'calib.txt'; target.write_bytes(b'incorrect')
        with self.assertRaisesRegex(ValueError, 'not overwritten'):
            downloader.verify_extract(self.asset(data), path, self.extracted, self.state, [])
        self.assertEqual(target.read_bytes(), b'incorrect')

    def test_path_traversal_is_rejected_before_extraction(self):
        data = archive_bytes({'../escaped': b'x'})
        path = self.root / 'tiny.zip'; path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'Unsafe ZIP'):
            downloader.verify_extract(self.asset(data), path, self.extracted, self.state, [])
        self.assertFalse((self.root / 'escaped').exists())

    def test_existing_wrong_size_archive_is_preserved_without_network(self):
        path = self.root / 'tiny.zip'; path.write_bytes(b'old')
        with mock.patch.object(downloader, 'probe', side_effect=AssertionError('must not request')):
            with self.assertRaisesRegex(ValueError, 'wrong size'):
                downloader.download(self.asset(b'new longer'), self.root, self.state, 1.)
        self.assertEqual(path.read_bytes(), b'old')

    def test_rejected_resume_preserves_partial_and_does_not_publish(self):
        data = archive_bytes({'calib.txt': b'123'})
        partial = self.root / 'tiny.zip.part'; partial.write_bytes(data[:20])
        fake = mock.Mock(); fake.poll.return_value = 33; fake.returncode = 33
        with mock.patch.object(downloader, 'probe', return_value={'size': len(data), 'url': 'https://example.invalid/tiny.zip', 'etag': 'fixed'}), \
             mock.patch.object(downloader.subprocess, 'Popen', return_value=fake) as launch:
            with self.assertRaisesRegex(RuntimeError, 'curl exit 33'):
                downloader.download(self.asset(data), self.root, self.state, 1.)
        self.assertIn('--continue-at', launch.call_args.args[0])
        self.assertEqual(partial.read_bytes(), data[:20])
        self.assertFalse((self.root / 'tiny.zip').exists())

    def test_depth_extraction_stays_with_selected_drive(self):
        data = archive_bytes({'train/chosen/proj_depth/a.png': b'yes', 'train/other/proj_depth/b.png': b'no'})
        path = self.root / 'tiny.zip'; path.write_bytes(data)
        result = downloader.verify_extract(self.asset(data, 'derived_depth'), path, self.extracted, self.state, ['chosen'])
        self.assertEqual(result['extracted_members'], 1)
        self.assertTrue((self.extracted / 'train/chosen/proj_depth/a.png').exists())
        self.assertFalse((self.extracted / 'train/other').exists())

    def test_verify_does_not_repair_missing_extracted_files(self):
        data = archive_bytes({'calib.txt': b'123'})
        path = self.root / 'tiny.zip'; path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'Extracted file missing'):
            downloader.verify_extract(self.asset(data), path, self.extracted, self.state, [], verify_only=True)
        self.assertFalse((self.extracted / 'calib.txt').exists())


if __name__ == '__main__':
    unittest.main()
