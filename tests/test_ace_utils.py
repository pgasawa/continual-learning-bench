import subprocess
import sys
import textwrap


def test_token_encoder_is_initialized_lazily_and_reused():
    code = textwrap.dedent(
        """
        import tiktoken

        calls = []

        class FakeEncoding:
            def encode(self, prompt):
                return prompt.split()

        def fake_get_encoding(name):
            calls.append(name)
            return FakeEncoding()

        tiktoken.get_encoding = fake_get_encoding

        from src.vendors.ace.utils import count_tokens

        assert calls == []
        assert count_tokens("one two") == 2
        assert count_tokens("three") == 1
        assert calls == ["cl100k_base"]
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
