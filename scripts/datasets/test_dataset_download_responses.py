"""Offline tests: no network requests and no dataset changes."""
import unittest

from geostreamfusion_setup_datasets import QuotaBlocked, check_response, source_request


class Response:
    def __init__(self, status, headers, body=b''):
        self.status_code, self.headers, self.body = status, headers, body

    def iter_content(self, _):
        yield self.body


class DownloadResponses(unittest.TestCase):
    def test_hf_source(self):
        url, params = source_request('hf:author/data@1234:sequences/file.zip')
        self.assertEqual(url, 'https://huggingface.co/datasets/author/data/resolve/1234/sequences/file.zip')
        self.assertEqual(params, {'download': 'true'})

    def test_unknown_source_rejected(self):
        with self.assertRaises(ValueError):
            source_request('unsupported:foo')

    def test_exact_binary_range(self):
        check_response(Response(206, {'Content-Range': 'bytes 10-19/100',
                                     'Content-Type': 'application/octet-stream'}), 10, 19, 100)

    def test_wrong_range_rejected(self):
        with self.assertRaises(RuntimeError):
            check_response(Response(206, {'Content-Range': 'bytes 0-9/100'}), 10, 19, 100)

    def test_quota_page_distinguished(self):
        with self.assertRaises(QuotaBlocked):
            check_response(Response(200, {'Content-Type': 'text/html'},
                                    b'<title>Google Drive - Quota exceeded</title>'), 0, 9, 100)

    def test_confirmation_is_not_archive(self):
        with self.assertRaisesRegex(RuntimeError, 'confirmation'):
            check_response(Response(200, {'Content-Type': 'text/html'},
                                    b'<html>confirm download</html>'), 0, 9, 100)


if __name__ == '__main__':
    unittest.main()
