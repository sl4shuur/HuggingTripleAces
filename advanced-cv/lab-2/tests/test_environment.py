import sys
from types import SimpleNamespace

import pytest

from alseg.runtime.environment import detect_environment, get_secret


def test_environment_detection_and_kaggle_priority(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.setattr("alseg.runtime.environment.Path.exists", lambda _: False)
    assert detect_environment() == "local"
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release")
    assert detect_environment() == "colab"
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert detect_environment() == "kaggle"


@pytest.mark.parametrize("environment", ["local", "kaggle", "colab"])
def test_secret_access_uses_the_selected_platform(monkeypatch, environment):
    secret_name = "WANDB_API_KEY"
    secret_value = "test-platform-key"
    monkeypatch.setenv(secret_name, secret_value)

    def read_secret(name):
        assert name == secret_name
        return secret_value

    monkeypatch.setitem(
        sys.modules,
        "kaggle_secrets",
        SimpleNamespace(
            UserSecretsClient=lambda: SimpleNamespace(get_secret=read_secret)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "google.colab",
        SimpleNamespace(userdata=SimpleNamespace(get=read_secret)),
    )
    assert get_secret(secret_name, environment=environment) == secret_value
