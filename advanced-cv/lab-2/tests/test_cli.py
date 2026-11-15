import json
from pathlib import Path

from click.testing import CliRunner

from setup_project import main


def test_setup_reports_runtime_without_requiring_dataset():
    config = Path(__file__).parents[1] / "configs" / "base.toml"
    result = CliRunner().invoke(main, ["--config", str(config)])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["data_checked"] is False
    assert report["python"]
    assert report["torch"]


def test_setup_help_lists_supported_options():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
    assert "--check-data" in result.output


def test_missing_voc_data_is_a_cli_error(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        f'[data]\ndataset_root = "{tmp_path.as_posix()}"\nvoc_root = ""\n'
    )
    result = CliRunner().invoke(main, ["--config", str(config), "--check-data"])
    assert result.exit_code == 1
    assert "VOC2012_test alone is insufficient" in result.output
    assert "Traceback" not in result.output
