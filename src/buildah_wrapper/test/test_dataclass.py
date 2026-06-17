import io
import pytest
from unittest.mock import patch, MagicMock

from buildah_wrapper.helper._dataclass import (
    ArgParser,
    BuildBuildah,
    BuildahBuildError,
)


def _make_build(**overrides):
    base = dict(
        service_name="svc",
        build_context="path/to/context",
        dockerfile="Dockerfile",
        image_name="docker.io/epicmorg/app:latest",
        build_args={},
    )
    base.update(overrides)
    return BuildBuildah(**base)


def _mock_process(stdout="", stderr="", rc=0):
    p = MagicMock()
    p.stdout = io.StringIO(stdout)
    p.stderr = io.StringIO(stderr)
    p.wait.return_value = rc
    return p


import sys


@pytest.fixture
def restore_sys_argv():
    original_argv = sys.argv
    yield
    sys.argv = original_argv


# --- ArgParser ---------------------------------------------------------------


def test_argparser_default_compose_file(restore_sys_argv, monkeypatch):
    monkeypatch.delenv("COMPOSE_FILE", raising=False)
    sys.argv = ["prog"]
    assert ArgParser().parse_args().compose_file == "docker-compose.yml"


def test_argparser_build_deploy_clean_flags(restore_sys_argv):
    sys.argv = ["prog", "--build", "--deploy"]
    args = ArgParser().parse_args()
    assert args.build is True
    assert args.deploy is True
    assert args.clean is False


def test_argparser_verbose_flag(restore_sys_argv):
    sys.argv = ["prog", "--verbose"]
    assert ArgParser().parse_args().verbose is True


def test_argparser_log_level(restore_sys_argv):
    sys.argv = ["prog", "--log-level", "DEBUG"]
    assert ArgParser().parse_args().log_level == "DEBUG"


def test_argparser_squash_default_false(restore_sys_argv):
    sys.argv = ["prog"]
    assert ArgParser().parse_args().squash is False


def test_argparser_squash(restore_sys_argv):
    sys.argv = ["prog", "--squash"]
    assert ArgParser().parse_args().squash is True


def test_argparser_no_squash(restore_sys_argv):
    sys.argv = ["prog", "--no-squash"]
    assert ArgParser().parse_args().squash is False


# --- _generate_build_command -------------------------------------------------


def test_generate_build_command_basics():
    cmd = _make_build(
        build_args={"ARG1": "value1", "ARG2": "value2"}
    )._generate_build_command()

    assert cmd[:2] == ["buildah", "build"]
    assert "--isolation=oci" in cmd
    assert "--cap-add=ALL" in cmd
    assert "--network=host" in cmd
    assert "--disable-compression=false" in cmd
    assert "--format" in cmd and "docker" in cmd
    assert "--no-cache" in cmd
    assert "--rm" in cmd
    assert "--layers=false" in cmd
    assert cmd.count("--build-arg") == 2
    assert "ARG1=value1" in cmd
    assert "ARG2=value2" in cmd
    # tag and dockerfile/context tail
    assert "-t" in cmd
    assert "docker.io/epicmorg/app:latest" in cmd
    assert "-f" in cmd
    assert cmd[-1] == "path/to/context"


def test_generate_build_command_squash_off_by_default():
    assert "--squash" not in _make_build()._generate_build_command()


def test_generate_build_command_squash_on():
    assert "--squash" in _make_build(squash=True)._generate_build_command()


def test_generate_build_command_dockerfile_path():
    cmd = _make_build(dockerfile="Dockerfile.dev")._generate_build_command()
    assert "path/to/context/Dockerfile.dev" in cmd


def test_generate_build_command_no_network_when_unset():
    cmd = _make_build(network=None)._generate_build_command()
    assert not any(t.startswith("--network") for t in cmd)


# --- _push_command / _destinations -------------------------------------------


def test_push_command_uses_docker_transport():
    cmd = _make_build()._push_command("quay.io/epicmorg/app:latest")
    assert cmd == [
        "buildah",
        "push",
        "docker.io/epicmorg/app:latest",
        "docker://quay.io/epicmorg/app:latest",
    ]


def test_destinations_dedup_and_order():
    dests = _make_build(
        image_name="docker.io/epicmorg/app:latest",
        mirrors=[
            "quay.io/epicmorg/app:latest",
            "docker.io/epicmorg/app:latest",  # duplicate of primary -> dropped
            "  ",  # blank after strip -> dropped
        ],
    )._destinations()
    assert dests == [
        "docker.io/epicmorg/app:latest",
        "quay.io/epicmorg/app:latest",
    ]


# --- build() -----------------------------------------------------------------


@patch("buildah_wrapper.helper._dataclass.subprocess.Popen")
@patch("buildah_wrapper.helper._dataclass.os.path.exists", return_value=True)
def test_build_raises_on_nonzero_returncode(mock_exists, mock_popen):
    mock_popen.return_value = _mock_process(stderr="error: build failed\n", rc=1)
    with pytest.raises(BuildahBuildError):
        _make_build().build()


@patch("buildah_wrapper.helper._dataclass.subprocess.Popen")
@patch("buildah_wrapper.helper._dataclass.os.path.exists", return_value=True)
def test_build_succeeds_on_zero_returncode(mock_exists, mock_popen):
    mock_popen.return_value = _mock_process(stdout="STEP 1/3\nSTEP 2/3\n", rc=0)
    _make_build().build()  # should not raise


def test_build_raises_when_context_missing():
    with pytest.raises(FileNotFoundError):
        _make_build(build_context="definitely/missing/path").build()


# --- deploy() ----------------------------------------------------------------


@patch("buildah_wrapper.helper._dataclass.subprocess.Popen")
def test_deploy_pushes_primary_and_mirrors(mock_popen):
    mock_popen.side_effect = [_mock_process(rc=0), _mock_process(rc=0)]
    _make_build(mirrors=["quay.io/epicmorg/app:latest"]).deploy()
    assert mock_popen.call_count == 2  # primary + one mirror


@patch("buildah_wrapper.helper._dataclass.subprocess.Popen")
def test_deploy_raises_on_push_failure(mock_popen):
    mock_popen.side_effect = [_mock_process(rc=0), _mock_process(rc=1)]
    with pytest.raises(BuildahBuildError):
        _make_build(mirrors=["quay.io/epicmorg/app:latest"]).deploy()


def test_deploy_raises_without_image_name():
    with pytest.raises(ValueError):
        _make_build(image_name="").deploy()