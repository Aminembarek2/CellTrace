"""Explicit credential loading preserves environment precedence and rejects expressions."""

from pathlib import Path

import pytest

from celltrace.secrets import load_openrouter_key


def test_credential_loading_reads_only_the_literal_key(tmp_path: Path) -> None:
    credential_file = tmp_path / ".env"
    credential_file.write_text(
        'export OPENROUTER_API_KEY="local_mock_key"\nCELLTRACE_MODEL=unexpected_model\n'
    )
    environment: dict[str, str] = {}
    load_openrouter_key(credential_file, environment)
    assert environment == {"OPENROUTER_API_KEY": "local_mock_key"}
    environment["OPENROUTER_API_KEY"] = "existing_key"
    load_openrouter_key(tmp_path / "missing", environment)
    assert environment["OPENROUTER_API_KEY"] == "existing_key"


@pytest.mark.parametrize(
    "contents",
    [
        "OPENROUTER_API_KEY=$(touch marker)\n",
        "OPENROUTER_API_KEY=`touch marker`\n",
        "OPENROUTER_API_KEY=first\nOPENROUTER_API_KEY=second\n",
        "OTHER_KEY=irrelevant\n",
    ],
)
def test_invalid_credentials_fail_without_disclosing_contents(
    tmp_path: Path, contents: str
) -> None:
    credential_file = tmp_path / ".env"
    credential_file.write_text(contents)
    environment: dict[str, str] = {}
    with pytest.raises(ValueError) as raised:
        load_openrouter_key(credential_file, environment)
    assert contents.strip() not in str(raised.value)
    assert environment == {}
    assert not (tmp_path / "marker").exists()
