import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

pytest.importorskip("requests")
from requests import Response
from requests.exceptions import HTTPError

from script.publish_source import main, publish_release, stage_source

DATASET = "denysmykhailov/alseg-source"


def http_error(status):
    response = Response()
    response.status_code = status
    return HTTPError(response=response)


class FakeApi:
    def __init__(self, status="ready", version="0.4.0"):
        self.status = status
        self.version = version
        self.calls = []
        self.error = None
        self.result_status = "ok"
        self.missing_manifest = False

    def dataset_status(self, dataset):
        assert dataset == DATASET
        if isinstance(self.status, Exception):
            raise self.status
        return self.status

    def dataset_download_file(self, dataset, file_name, path, quiet, unzip):
        assert file_name == "source-release.json" and unzip
        if self.missing_manifest:
            raise http_error(404)
        (Path(path) / file_name).write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "project_version": self.version,
                }
            )
        )

    def dataset_create_new(self, folder, **kwargs):
        self.calls.append(("create", kwargs))
        return SimpleNamespace(error=self.error, status=self.result_status)

    def dataset_create_version(self, folder, **kwargs):
        self.calls.append(("update", kwargs))
        return SimpleNamespace(error=self.error, status=self.result_status)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "project"
    (root / "alseg").mkdir(parents=True)
    (root / "script").mkdir()
    (root / "configs").mkdir()
    (root / "pyproject.toml").write_text('[project]\nversion = "0.5.0"\n')
    (root / "alseg" / "data.py").write_text('MESSAGE = "source"\n')
    (root / "script" / "entry.py").write_text('print("entry")\n')
    (root / "configs" / "base.toml").write_text("seed = 42\n")
    return root


@pytest.mark.parametrize("notebook_name", ["setup.ipynb", "train.ipynb"])
def test_bundle_is_source_only_with_clean_notebook_and_hashes(
    source, tmp_path, notebook_name
):
    (source / ".env").write_text("TOKEN=private")
    (source / "kaggle.json").write_text("private")
    (source / "alseg" / "secrets.py").write_text('TOKEN = "private"')
    (source / "dist").mkdir()
    (source / "dist" / "model.pt").write_bytes(b"weights")
    (source / "kaggle").mkdir()
    (source / "kaggle" / notebook_name).write_text(
        json.dumps(
            {
                "metadata": {"private": "do not upload"},
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["print(1)"],
                        "outputs": [{"text": "private"}],
                        "metadata": {},
                        "execution_count": 1,
                    }
                ],
            }
        )
    )
    output = tmp_path / "bundle"
    release = stage_source(source, output, DATASET, "ALSeg source")
    with zipfile.ZipFile(output / "source.zip") as archive:
        assert set(archive.namelist()) == set(release["files"])
        assert ".env" not in archive.namelist()
        assert "kaggle.json" not in archive.namelist()
        assert "alseg/secrets.py" not in archive.namelist()
        assert "dist/model.pt" not in archive.namelist()
        notebook = json.loads(archive.read(f"kaggle/{notebook_name}"))
        assert notebook["cells"][0]["outputs"] == []
        assert notebook["cells"][0]["execution_count"] is None
        assert notebook["metadata"] == {}
        for name, digest in release["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    assert (
        release["archive_sha256"]
        == hashlib.sha256((output / "source.zip").read_bytes()).hexdigest()
    )


def test_reject_server_url_and_source_symlink(source, tmp_path):
    server_url = (
        "https://kkb-production.jupyter-proxy." + "kaggle.net/k/access-token/proxy"
    )
    file = source / "alseg" / "data.py"
    file.write_text(server_url)
    with pytest.raises(ValueError, match="server access URL"):
        stage_source(source, tmp_path / "bundle", DATASET, "ALSeg source")
    file.unlink()
    outside = tmp_path / "outside.py"
    outside.write_text("private")
    file.symlink_to(outside)
    with pytest.raises(ValueError, match="symlinks"):
        stage_source(source, tmp_path / "bundle", DATASET, "ALSeg source")


def test_create_only_on_404_and_private_by_default(tmp_path):
    api = FakeApi(status=http_error(404))
    release = {"dataset": DATASET, "project_version": "0.5.0"}
    assert publish_release(api, tmp_path, release) == "created"
    assert api.calls[0][1]["public"] is False
    assert api.calls[0][1]["convert_to_csv"] is False
    forbidden = FakeApi(status=http_error(403))
    with pytest.raises(ValueError, match="403.*--create"):
        publish_release(forbidden, tmp_path, release)
    assert forbidden.calls == []


def test_explicit_create_bypasses_forbidden_status_check(tmp_path):
    api = FakeApi(status=http_error(403))
    release = {"dataset": DATASET, "project_version": "0.5.0"}
    assert publish_release(api, tmp_path, release, create=True) == "created"
    assert api.calls == [
        ("create", {"public": False, "quiet": True, "convert_to_csv": False})
    ]


def test_authentication_error_does_not_create_a_dataset(tmp_path):
    api = FakeApi(status=http_error(401))
    with pytest.raises(HTTPError):
        publish_release(api, tmp_path, {"dataset": DATASET, "project_version": "0.5.0"})
    assert api.calls == []


@pytest.mark.parametrize("version", ["0.5.0", "0.6.0"])
def test_update_rejects_same_or_older_local_version(tmp_path, version):
    api = FakeApi(version=version)
    with pytest.raises(ValueError, match="Increase pyproject.toml"):
        publish_release(api, tmp_path, {"dataset": DATASET, "project_version": "0.5.0"})
    assert api.calls == []


def test_update_preserves_old_versions_and_uses_new_version_notes(tmp_path):
    api = FakeApi()
    release = {"dataset": DATASET, "project_version": "0.5.0"}
    assert publish_release(api, tmp_path, release) == "updated"
    assert api.calls == [
        (
            "update",
            {
                "version_notes": "ALSeg 0.5.0",
                "quiet": True,
                "convert_to_csv": False,
                "delete_old_versions": False,
            },
        )
    ]


def test_refuse_unmanaged_or_processing_dataset_and_surface_api_error(tmp_path):
    release = {"dataset": DATASET, "project_version": "0.5.0"}
    api = FakeApi()
    api.missing_manifest = True
    with pytest.raises(ValueError, match="refusing to overwrite"):
        publish_release(api, tmp_path, release)
    assert api.calls == []
    processing = FakeApi(status="reprocessing")
    with pytest.raises(ValueError, match="wait until"):
        publish_release(processing, tmp_path, release)
    assert processing.calls == []
    rejected = FakeApi(status=http_error(404))
    rejected.error = "duplicate slug"
    with pytest.raises(ValueError, match="duplicate slug"):
        publish_release(rejected, tmp_path, release)


def test_non_ok_response_without_error_text_is_not_reported_as_success(tmp_path):
    api = FakeApi(status=http_error(404))
    api.result_status = "error"
    with pytest.raises(ValueError, match="Kaggle rejected"):
        publish_release(api, tmp_path, {"dataset": DATASET, "project_version": "0.5.0"})


def test_dry_run_creates_bundle_without_kaggle_authentication(source, tmp_path):
    output = tmp_path / "bundle"
    result = CliRunner().invoke(
        main,
        [
            "--dataset",
            DATASET,
            "--project-dir",
            str(source),
            "--output-dir",
            str(output),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "project 0.5.0" in result.output
    assert (output / "source.zip").is_file()
