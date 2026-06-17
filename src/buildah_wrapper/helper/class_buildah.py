import shutil
import subprocess
from collections import defaultdict

from buildah_wrapper.helper.log_print import logger
from buildah_wrapper.helper._dataclass import (
    ComposeFileLoader,
    BuildBuildah,
    BuildahBuildError,
)


class BuildahBuilder:
    """Class responsible for orchestrating build / deploy / clean phases."""

    def __init__(self, args):
        self.args = args
        self.compose_file = args.compose_file
        self.do_build = args.build
        self.do_deploy = args.deploy
        self.do_clean = args.clean
        self.squash_default = args.squash
        self.compose_data = {}
        self.services = []

    def preflight(self):
        """Fail fast if the buildah binary is missing or broken.

        buildah runs as a native host binary (no executor image to pre-pull),
        so the equivalent up-front check is that the binary exists and answers.
        """
        if shutil.which("buildah") is None:
            raise RuntimeError("buildah binary not found in PATH")
        result = subprocess.run(
            ["buildah", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"buildah --version failed (rc={result.returncode}): "
                f"{result.stderr.strip()}"
            )
        logger.info(f"Using {result.stdout.strip()}")

    def validate_compose_file(self):
        """Validate and load the docker-compose file."""
        loader = ComposeFileLoader(self.compose_file)
        try:
            self.compose_data = loader.load()
            logger.info(f"Successfully loaded compose file: {self.compose_file}")
        except Exception as e:
            logger.error(f"Error loading compose file: {e}")
            raise e

    def process_services(self):
        """Process services from the docker-compose file into BuildBuildah."""
        if "services" not in self.compose_data:
            logger.error("No services found in docker-compose file.")
            raise ValueError("No services found in docker-compose file.")

        image_names = defaultdict(int)

        for service_name, service_info in self.compose_data["services"].items():
            if not service_info:
                logger.warning(f"Skipping empty service definition: {service_name}")
                continue

            if "build" not in service_info:
                # No build section: nothing for buildah to build or push from
                # the local store.
                logger.info(f"Skipping service without build section: {service_name}")
                continue

            build = service_info["build"]
            if not isinstance(build, dict):
                # Short-form `build: ./path` is not handled by the command
                # builder; fail loudly instead of crashing on .get().
                raise ValueError(
                    f"{service_name}: short-form 'build: <path>' is not supported; "
                    f"use the long form with context/dockerfile."
                )

            dockerfile = build.get("dockerfile", "Dockerfile")
            build_context = build.get("context", ".")
            build_args = build.get("args", {})
            image_name = service_info.get("image", "")

            if not image_name:
                logger.warning(f"No image specified for service {service_name}")
                continue

            image_names[image_name] += 1

            mirrors = service_info.get("x-mirrors", []) or []
            if not isinstance(mirrors, list):
                raise ValueError(
                    f"{service_name}: x-mirrors must be a list of image "
                    f"references, got {type(mirrors).__name__}"
                )

            squash = service_info.get("x-squash", self.squash_default)
            if not isinstance(squash, bool):
                raise ValueError(
                    f"{service_name}: x-squash must be a boolean (true/false), "
                    f"got {type(squash).__name__}"
                )

            self.services.append(
                BuildBuildah(
                    service_name=service_name,
                    build_context=build_context,
                    dockerfile=dockerfile,
                    image_name=image_name,
                    build_args=build_args,
                    squash=squash,
                    mirrors=mirrors,
                )
            )

        # A reused image name across services overwrites the same store entry;
        # fail loudly instead of silently clobbering.
        duplicates = [name for name, count in image_names.items() if count > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate image name(s) across services: {', '.join(duplicates)}"
            )

    def build_services(self):
        """Build all services. Aggregate failures, exit non-zero if any fail."""
        if not self.services:
            logger.warning("No services to build.")
            return

        failed = []
        for service in self.services:
            try:
                service.build()
            except Exception as e:
                logger.error(f"Failed to build service {service.service_name}: {e}")
                failed.append(service.service_name)

        if failed:
            raise BuildahBuildError(
                f"{len(failed)} service(s) failed to build: {', '.join(failed)}"
            )

    def deploy_services(self):
        """Push all services. Aggregate failures, exit non-zero if any fail."""
        if not self.services:
            logger.warning("No services to deploy.")
            return

        failed = []
        for service in self.services:
            try:
                service.deploy()
            except Exception as e:
                logger.error(f"Failed to deploy service {service.service_name}: {e}")
                failed.append(service.service_name)

        if failed:
            raise BuildahBuildError(
                f"{len(failed)} service(s) failed to deploy: {', '.join(failed)}"
            )

    def clean_store(self):
        """Remove all buildah containers and images (buildah-only phase)."""
        for label, command in (
            ("containers", ["buildah", "rm", "--all"]),
            ("images", ["buildah", "rmi", "--all"]),
        ):
            logger.info(f"Cleaning Buildah {label}: {' '.join(command)}")
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for line in result.stdout.splitlines():
                if line.strip():
                    logger.info(line.strip())
            if result.returncode != 0:
                for line in result.stderr.splitlines():
                    if line.strip():
                        logger.error(line.strip())
                raise BuildahBuildError(
                    f"Failed to clean Buildah {label} (rc={result.returncode})"
                )
        logger.info("Successfully cleaned all Buildah containers and images.")

    def run(self):
        """Orchestrate the requested phases.

        Phases are sequential and ordered: build the whole graph first, then
        deploy. A build failure raises before any push, so a half-built graph
        is never partially pushed. Sequential build (no naive parallelism)
        respects FROM chains like light -> main -> jdk17 -> jira.
        """
        self.preflight()

        if self.do_clean:
            self.clean_store()
            return

        self.validate_compose_file()
        self.process_services()

        if self.do_build:
            self.build_services()
        if self.do_deploy:
            self.deploy_services()