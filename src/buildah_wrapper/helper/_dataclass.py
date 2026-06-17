import os
import yaml
import shutil
import argparse
import subprocess
import threading
from collections import deque
from typing import Dict, List, Optional

from dataclasses import dataclass, field
from buildah_wrapper.helper.log_print import logger


class BuildahBuildError(RuntimeError):
    """Raised when a buildah build or push returns a non-zero exit code."""


@dataclass
class ComposeFileLoader:
    """Class responsible for loading the docker-compose.yml file."""

    compose_file: str

    def load(self) -> Dict:
        """Load and parse the YAML docker-compose file."""
        if not os.path.exists(self.compose_file):
            raise FileNotFoundError(f"The file {self.compose_file} does not exist.")
        try:
            with open(self.compose_file, "r") as file:
                return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise Exception(f"Error loading YAML file: {e}")


@dataclass
class ArgParser:
    """Class responsible for parsing command line arguments."""

    parser: argparse.ArgumentParser = None

    def __post_init__(self):
        if self.parser is None:
            self.parser = argparse.ArgumentParser(
                description="Buildah-Compose Wrapper", add_help=False
            )
            self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument(
            "--compose-file",
            default=os.getenv("COMPOSE_FILE", "docker-compose.yml"),
            help="Path to docker-compose.yml file",
        )
        # Actions are combinable: --build --deploy builds then pushes in one run.
        self.parser.add_argument(
            "--build",
            "-b",
            action="store_true",
            help="Build images into the local store",
        )
        self.parser.add_argument(
            "--deploy",
            "-d",
            action="store_true",
            help="Push built images to the registry and all x-mirrors",
        )
        self.parser.add_argument(
            "--clean",
            action="store_true",
            help="Remove all buildah containers and images",
        )
        self.parser.add_argument(
            "--version", "-v", action="store_true", help="Show script version"
        )
        self.parser.add_argument(
            "--help", "-h", action="store_true", help="Show this help message and exit"
        )
        self.parser.add_argument(
            "--verbose",
            "-V",
            action="store_true",
            help="Verbose output (shortcut for --log-level DEBUG)",
        )
        self.parser.add_argument(
            "--log-level",
            default=None,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Override log level (default: from settings / LOG_LEVEL env)",
        )
        self.parser.add_argument(
            "--squash",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Single-layer output for all services (default: off). "
                 "Per-service x-squash overrides this.",
        )

    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()


@dataclass
class BuildBuildah:
    """Class responsible for building and pushing one image with Buildah."""

    service_name: str
    build_context: str
    dockerfile: str
    image_name: str
    build_args: Dict[str, str]
    squash: bool = False
    # Extra push destinations, populated from the compose `x-mirrors` key.
    mirrors: List[str] = field(default_factory=list)

    # buildah build flags, exposed as fields with the project's working
    # defaults instead of being hardcoded in the command body. Tune here, not
    # in _generate_build_command.
    isolation: str = "oci"
    cap_add: List[str] = field(default_factory=lambda: ["ALL"])
    # Network for RUN steps during the build (not a run-container network).
    network: Optional[str] = "host"
    # OCI format does not carry HEALTHCHECK; docker format does -> keep docker.
    image_format: str = "docker"
    layers: bool = False
    no_cache: bool = True
    disable_compression: bool = False
    rm: bool = True

    def build(self) -> None:
        """Build the image into the local store."""
        if not os.path.exists(self.build_context):
            raise FileNotFoundError(f"Build context not found: {self.build_context}")
        dockerfile_path = os.path.join(self.build_context, self.dockerfile)
        if not os.path.exists(dockerfile_path):
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

        command = self._generate_build_command()
        logger.info(
            f"Building {self.service_name} with Buildah: {' '.join(command)}"
        )

        rc = self._run_streamed(command, self.service_name)
        if rc != 0:
            raise BuildahBuildError(
                f"{self.service_name}: buildah build exited with code {rc}"
            )
        logger.info(f"{self.service_name} built successfully.")

    def deploy(self) -> None:
        """Push the built image to the primary registry and every mirror.

        buildah has no kaniko-style multi-destination: build lands the image in
        the local store, then each destination is a separate `buildah push`.
        A failed push to any destination is fatal for the service (the mirror
        failure stops the remaining pushes and raises), mirroring the kaniko
        "mirror failure fails the build" contract.
        """
        destinations = self._destinations()
        if not destinations:
            raise ValueError(
                f"{self.service_name}: no image name to push (set `image:`)."
            )

        for dest in destinations:
            command = self._push_command(dest)
            logger.info(
                f"Pushing {self.service_name} -> {dest}: {' '.join(command)}"
            )
            rc = self._run_streamed(command, f"{self.service_name} push {dest}")
            if rc != 0:
                raise BuildahBuildError(
                    f"{self.service_name}: push to {dest} exited with code {rc}"
                )

        logger.info(
            f"{self.service_name} pushed -> {', '.join(destinations)}"
        )

    def _generate_build_command(self) -> List[str]:
        """Generate the `buildah build` command from the configured fields."""
        command = ["buildah", "build", f"--isolation={self.isolation}"]
        for cap in self.cap_add:
            command.append(f"--cap-add={cap}")
        if self.network:
            command.append(f"--network={self.network}")
        command.append(
            f"--disable-compression={'true' if self.disable_compression else 'false'}"
        )
        command += ["--format", self.image_format]
        if self.no_cache:
            command.append("--no-cache")
        if self.rm:
            command.append("--rm")
        command.append(f"--layers={'true' if self.layers else 'false'}")
        if self.squash:
            command.append("--squash")

        for arg_name, arg_value in self.build_args.items():
            command.extend(["--build-arg", f"{arg_name}={arg_value}"])

        command += [
            "-f",
            os.path.join(self.build_context, self.dockerfile),
            "-t",
            self.image_name,
            self.build_context,
        ]
        return command

    def _push_command(self, dest: str) -> List[str]:
        """Push the locally built image to an explicit docker:// destination."""
        return ["buildah", "push", self.image_name, f"docker://{dest}"]

    def _destinations(self) -> List[str]:
        """Primary image plus mirrors, de-duplicated, empties dropped."""
        seen = set()
        result: List[str] = []
        for dest in [self.image_name, *self.mirrors]:
            dest = (dest or "").strip()
            if dest and dest not in seen:
                seen.add(dest)
                result.append(dest)
        return result

    def _run_streamed(self, command: List[str], label: str) -> int:
        """Run a command, draining both pipes concurrently, return its rc.

        Reading stdout to EOF before touching stderr deadlocks on heavy images:
        buildah writes progress to stderr, fills the 64K kernel pipe buffer,
        blocks on write, and stops feeding stdout -> the reader blocks forever
        and wait() is never reached. Draining both in parallel removes that.
        Severity is not inferred from the stream: stdout -> info, stderr ->
        debug. Success/failure is decided solely by the return code.
        """
        stderr_tail: deque = deque(maxlen=50)

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        t_out = threading.Thread(
            target=self._drain, args=(process.stdout, logger.info), daemon=True
        )
        t_err = threading.Thread(
            target=self._drain,
            args=(process.stderr, logger.debug, stderr_tail),
            daemon=True,
        )
        t_out.start()
        t_err.start()
        t_out.join()
        t_err.join()

        rc = process.wait()
        if rc != 0:
            for line in stderr_tail:
                logger.error(f"[{label}] {line}")
        return rc

    @staticmethod
    def _drain(stream, log_fn, sink: Optional[deque] = None) -> None:
        """Read a child stream line by line, log each line, optionally buffer."""
        try:
            for raw in iter(stream.readline, ""):
                line = raw.rstrip("\n")
                if not line:
                    continue
                log_fn(line)
                if sink is not None:
                    sink.append(line)
        finally:
            stream.close()
