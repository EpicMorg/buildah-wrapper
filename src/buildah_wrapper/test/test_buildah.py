import os
import yaml
import pytest
from unittest.mock import patch, MagicMock

from buildah_wrapper.helper.class_buildah import (
    ComposeFileLoader,
    BuildahBuilder,
    BuildBuildah,
)
from buildah_wrapper.helper._dataclass import BuildahBuildError


def _builder(**args_overrides):
    args = MagicMock()
    args.compose_file = "docker-compose.yml"
    args.build = True
    args.deploy = False
    args.clean = False
    args.squash = False
    for k, v in args_overrides.items():
        setattr(args, k, v)
    return BuildahBuilder(args)


# --- ComposeFileLoader -------------------------------------------------------


@pytest.fixture
def create_test_file():
    test_file_path = "test_compose_file.yaml"
    with open(test_file_path, "w") as file:
        yaml.dump({"key": "value"}, file)
    yield test_file_path
    if os.path.exists(test_file_path):
        os.remove(test_file_path)


def test_load_raises_when_file_does_not_exist():
    loader = ComposeFileLoader("non_existent_compose_file.yaml")
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_load_returns_data_when_file_exists(create_test_file):
    assert ComposeFileLoader(create_test_file).load() == {"key": "value"}


# --- process_services --------------------------------------------------------


def test_process_services():
    b = _builder()
    b.compose_data = {
        "services": {
            "s1": {"build": {"dockerfile": "Dockerfile1", "context": "c1", "args": {"A": "1"}}, "image": "img1"},
            "s2": {"build": {"dockerfile": "Dockerfile2", "context": "c2", "args": {"B": "2"}}, "image": "img2"},
        }
    }
    b.process_services()
    assert len(b.services) == 2
    assert all(isinstance(s, BuildBuildah) for s in b.services)
    assert [s.service_name for s in b.services] == ["s1", "s2"]


@patch("buildah_wrapper.helper.class_buildah.logger")
def test_process_services_no_services(mock_logger):
    b = _builder()
    b.compose_data = {}
    with pytest.raises(ValueError):
        b.process_services()
    mock_logger.error.assert_called_once_with("No services found in docker-compose file.")


def test_process_services_skips_empty_no_build_and_no_image():
    b = _builder()
    b.compose_data = {
        "services": {
            "empty": None,                       # empty definition -> skip
            "external": {"image": "redis:7"},    # no build section -> skip
            "noimg": {"build": {"context": "."}},  # no image -> skip
            "app": {"build": {"context": "."}, "image": "app:latest"},
        }
    }
    b.process_services()
    assert [s.service_name for s in b.services] == ["app"]


def test_process_services_short_form_build_rejected():
    b = _builder()
    b.compose_data = {"services": {"app": {"build": "./app", "image": "app:latest"}}}
    with pytest.raises(ValueError):
        b.process_services()


def test_process_services_duplicate_image_names_rejected():
    b = _builder()
    b.compose_data = {
        "services": {
            "a": {"build": {"context": "."}, "image": "same:latest"},
            "b": {"build": {"context": "."}, "image": "same:latest"},
        }
    }
    with pytest.raises(ValueError):
        b.process_services()


# --- x-mirrors ---------------------------------------------------------------


def test_process_services_parses_x_mirrors():
    b = _builder()
    b.compose_data = {
        "services": {
            "app": {
                "build": {"context": "."},
                "image": "docker.io/epicmorg/app:latest",
                "x-mirrors": ["quay.io/epicmorg/app:latest"],
            }
        }
    }
    b.process_services()
    assert b.services[0].mirrors == ["quay.io/epicmorg/app:latest"]


def test_process_services_x_mirrors_must_be_list():
    b = _builder()
    b.compose_data = {
        "services": {
            "app": {"build": {"context": "."}, "image": "app:latest", "x-mirrors": "quay.io/app:latest"}
        }
    }
    with pytest.raises(ValueError):
        b.process_services()


# --- x-squash ----------------------------------------------------------------


def test_process_services_squash_cli_default_false():
    b = _builder(squash=False)
    b.compose_data = {"services": {"app": {"build": {"context": "."}, "image": "app:latest"}}}
    b.process_services()
    assert b.services[0].squash is False


def test_process_services_squash_cli_default_true():
    b = _builder(squash=True)
    b.compose_data = {"services": {"app": {"build": {"context": "."}, "image": "app:latest"}}}
    b.process_services()
    assert b.services[0].squash is True


def test_process_services_x_squash_overrides_cli():
    b = _builder(squash=False)
    b.compose_data = {"services": {"app": {"build": {"context": "."}, "image": "app:latest", "x-squash": True}}}
    b.process_services()
    assert b.services[0].squash is True


def test_process_services_x_squash_false_overrides_cli_true():
    b = _builder(squash=True)
    b.compose_data = {"services": {"app": {"build": {"context": "."}, "image": "app:latest", "x-squash": False}}}
    b.process_services()
    assert b.services[0].squash is False


def test_process_services_x_squash_must_be_bool():
    b = _builder()
    b.compose_data = {"services": {"app": {"build": {"context": "."}, "image": "app:latest", "x-squash": "true"}}}
    with pytest.raises(ValueError):
        b.process_services()


# --- preflight ---------------------------------------------------------------


@patch("buildah_wrapper.helper.class_buildah.shutil.which", return_value=None)
def test_preflight_missing_binary(mock_which):
    with pytest.raises(RuntimeError):
        _builder().preflight()


@patch("buildah_wrapper.helper.class_buildah.subprocess.run")
@patch("buildah_wrapper.helper.class_buildah.shutil.which", return_value="/usr/bin/buildah")
def test_preflight_ok(mock_which, mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="buildah version 1.37.0", stderr="")
    _builder().preflight()  # should not raise


@patch("buildah_wrapper.helper.class_buildah.subprocess.run")
@patch("buildah_wrapper.helper.class_buildah.shutil.which", return_value="/usr/bin/buildah")
def test_preflight_version_failure(mock_which, mock_run):
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="boom")
    with pytest.raises(RuntimeError):
        _builder().preflight()


# --- build_services / deploy_services ----------------------------------------


def test_build_services_runs_all():
    s1, s2 = MagicMock(), MagicMock()
    b = _builder()
    b.services = [s1, s2]
    b.build_services()
    s1.build.assert_called_once()
    s2.build.assert_called_once()


def test_build_services_aggregates_failures():
    s1 = MagicMock(service_name="s1")
    s1.build.side_effect = BuildahBuildError("boom")
    s2 = MagicMock(service_name="s2")
    b = _builder()
    b.services = [s1, s2]
    with pytest.raises(BuildahBuildError):
        b.build_services()
    # aggregate-continue: s2 still attempted after s1 fails
    s1.build.assert_called_once()
    s2.build.assert_called_once()


def test_deploy_services_aggregates_failures():
    s1 = MagicMock(service_name="s1")
    s1.deploy.side_effect = BuildahBuildError("boom")
    s2 = MagicMock(service_name="s2")
    b = _builder()
    b.services = [s1, s2]
    with pytest.raises(BuildahBuildError):
        b.deploy_services()
    s1.deploy.assert_called_once()
    s2.deploy.assert_called_once()


# --- clean_store -------------------------------------------------------------


@patch("buildah_wrapper.helper.class_buildah.subprocess.run")
def test_clean_store_runs_rm_and_rmi(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    _builder().clean_store()
    assert mock_run.call_count == 2  # buildah rm --all + buildah rmi --all


@patch("buildah_wrapper.helper.class_buildah.subprocess.run")
def test_clean_store_raises_on_failure(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="cannot remove")
    with pytest.raises(BuildahBuildError):
        _builder().clean_store()