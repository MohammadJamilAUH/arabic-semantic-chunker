import json
from pathlib import Path

from arabic_chunker.cli import main

SAMPLE = Path(__file__).parent.parent / "examples" / "sample.md"


def test_cli_json(capsys):
    assert main([str(SAMPLE), "--format", "json", "--max-tokens", "60", "--min-tokens", "10"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data and {"text", "start", "end", "headings", "contextual_text"} <= data[0].keys()


def test_cli_hierarchical(capsys):
    assert main([str(SAMPLE), "--hierarchical", "150", "--max-tokens", "40",
                 "--min-tokens", "10"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["parents"] and data["children"]
