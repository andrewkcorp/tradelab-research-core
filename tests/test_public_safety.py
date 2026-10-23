"""Regression tests for the bundled *heuristic* publication checker."""
from pathlib import Path
import tempfile
import unittest

from tools.check_public_tree import audit_tree,content_findings,path_findings


class PublicSafety(unittest.TestCase):
    def test_secret_match_is_redacted(self):
        invented = ('sk-'+'x'*40).encode()
        findings = content_findings(invented,'synthetic-test')
        self.assertEqual(findings[0]['rule'],'openai_key')
        self.assertNotIn(invented.decode(),str(findings))
        self.assertIn('fingerprint',findings[0])

    def test_home_account_and_key_paths_are_flagged(self):
        path = ('/Us'+'ers/'+'invented-user/project').encode()
        self.assertEqual(content_findings(path,'example')[0]['rule'],'personal_home_path')
        account = ('DE'+'MO'+'1234567').encode()
        self.assertEqual(content_findings(account,'example')[0]['rule'],'broker_account_identifier')
        self.assertTrue(path_findings('.env'))
        self.assertTrue(path_findings('data/vendor.csv'))
        self.assertTrue(path_findings('runs/cache.sqlite3'))

    def test_clean_text_and_generated_bytecode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'README.md').write_text('Synthetic research example.\n')
            (root/'__pycache__').mkdir()
            (root/'__pycache__/module.pyc').write_bytes(b'not an exportable source')
            report = audit_tree(root)
            self.assertEqual(report['files_scanned'],1)
            self.assertEqual(report['findings'],[])

    def test_symlink_is_flagged_and_target_is_not_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'public';root.mkdir()
            secret = Path(tmp)/'private.txt';secret.write_text('do not read')
            try: (root/'link').symlink_to(secret)
            except (OSError,NotImplementedError): self.skipTest('Symlinks unavailable')
            report = audit_tree(root)
            self.assertEqual(report['files_scanned'],0)
            self.assertEqual(report['findings'][0]['rule'],'symlink_not_allowed')


if __name__ == '__main__': unittest.main()
