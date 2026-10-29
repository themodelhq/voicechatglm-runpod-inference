"""Small CPU-only tests for input validation helpers; does not load the model."""
import unittest


class ValidationContractTests(unittest.TestCase):
    def test_expected_default_model_revision(self):
        from handler import MODEL_ID, MODEL_REVISION
        self.assertEqual(MODEL_ID, "themodelhq/voicechatglm-qwen3-8b")
        self.assertEqual(MODEL_REVISION, "9f25d05035bfda0de9b36dfc5911ddcdb378f3c8")

    def test_constant_time_compare(self):
        from handler import _constant_time_equal
        self.assertTrue(_constant_time_equal("secret", "secret"))
        self.assertFalse(_constant_time_equal("secret", "other"))


if __name__ == "__main__":
    unittest.main()
