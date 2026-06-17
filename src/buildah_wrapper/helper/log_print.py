import sys
import subprocess

from termcolor import colored
from dotenv import load_dotenv

from buildah_wrapper.settings import SCRIPT_VERSION
from buildah_wrapper.helper.helper import setup_logger

logger = setup_logger()
load_dotenv()


# Original EpicMorg Buildah Wrapper banner, preserved verbatim from 0.0.0.8.
ASCII_ART = rf"""
+=========================================================================+
 /$$$$$$$$         /$$         /$$      /$$
| $$_____/        |__/        | $$$    /$$$
| $$       /$$$$$$ /$$ /$$$$$$| $$$$  /$$$$ /$$$$$$  /$$$$$$  /$$$$$$
| $$$$$   /$$__  $| $$/$$_____| $$ $$/$$ $$/$$__  $$/$$__  $$/$$__  $$
| $$__/  | $$  \ $| $| $$     | $$  $$$| $| $$  \ $| $$  \__| $$  \ $$
| $$     | $$  | $| $| $$     | $$\  $ | $| $$  | $| $$     | $$  | $$
| $$$$$$$| $$$$$$$| $|  $$$$$$| $$ \/  | $|  $$$$$$| $$     |  $$$$$$$
|________| $$____/|__/\_______|__/     |__/\______/|__/      \____  $$
         | $$                                                /$$  \ $$
         | $$                                               |  $$$$$$/
 /$$$$$$$|__/      /$$/$$      /$$         /$$               \______/
| $$__  $$        |__| $$     | $$        | $$
| $$  \ $$/$$   /$$/$| $$ /$$$$$$$ /$$$$$$| $$$$$$$
| $$$$$$$| $$  | $| $| $$/$$__  $$|____  $| $$__  $$
| $$__  $| $$  | $| $| $| $$  | $$ /$$$$$$| $$  \ $$
| $$  \ $| $$  | $| $| $| $$  | $$/$$__  $| $$  | $$
| $$$$$$$|  $$$$$$| $| $|  $$$$$$|  $$$$$$| $$  | $$
|_______/ \______/|__|__/\_______/\_______|__/  |__/
 /$$      /$$
| $$  /$ | $$
| $$ /$$$| $$ /$$$$$$ /$$$$$$  /$$$$$$  /$$$$$$  /$$$$$$  /$$$$$$
| $$/$$ $$ $$/$$__  $|____  $$/$$__  $$/$$__  $$/$$__  $$/$$__  $$
| $$$$_  $$$| $$  \__//$$$$$$| $$  \ $| $$  \ $| $$$$$$$| $$  \__/
| $$$/ \  $$| $$     /$$__  $| $$  | $| $$  | $| $$_____| $$
| $$/   \  $| $$    |  $$$$$$| $$$$$$$| $$$$$$$|  $$$$$$| $$
|__/     \__|__/     \_______| $$____/| $$____/ \_______|__/
                             | $$     | $$
                             | $$     | $$
                             |__/     |__/
+=========================================================================+
"""


def get_buildah_version() -> str:
    """Return the installed buildah version string, or 'Unknown'."""
    try:
        result = subprocess.run(
            ["buildah", "--version"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Failed to get Buildah version: {e}")
        return "Unknown"


def show_help():
    """Display detailed information about available commands and arguments."""
    help_text = f"""{ASCII_ART}
{colored('EpicMorg Buildah Compose Wrapper', 'cyan', attrs=['bold'])}

This script builds and pushes images with Buildah, driven by a docker-compose.yml.

{colored('Commands:', 'yellow', attrs=['bold'])}
  {colored('--version, -v', 'green')}          : Show the version of the script and buildah
  {colored('--help, -h', 'green')}             : Display this help message

{colored('Actions (combinable, e.g. --build --deploy):', 'yellow', attrs=['bold'])}
  {colored('--build, -b', 'green')}            : Build images into the local store
  {colored('--deploy, -d', 'green')}           : Push built images to the registry and all x-mirrors
  {colored('--clean', 'green')}                : Remove all buildah containers and images

{colored('Options (CLI):', 'yellow', attrs=['bold'])}
  {colored('--compose-file FILE', 'green')}    : Path to the docker-compose.yml file (default: docker-compose.yml)
  {colored('--squash / --no-squash', 'green')} : Single-layer output, default OFF (x-squash overrides per service)
  {colored('--verbose, -V', 'green')}          : Verbose output (shortcut for --log-level DEBUG)
  {colored('--log-level LEVEL', 'green')}      : Override log level: DEBUG, INFO, WARNING, ERROR, CRITICAL

{colored('Mirrors (docker-compose.yml):', 'yellow', attrs=['bold'])}
  Add an {colored('x-mirrors', 'green')} list to a compose service to push the built image to
  additional registries during --deploy. A failed push to any mirror fails the
  service.

  services:
    app:
      image: docker.io/epicmorg/app:latest
      x-mirrors:
        - quay.io/epicmorg/app:latest

{colored('Squash (docker-compose.yml):', 'yellow', attrs=['bold'])}
  Add an {colored('x-squash', 'green')} field to a compose service to control single-layer
  output. Default is false. A per-service x-squash overrides the --squash CLI default.

  services:
    app:
      image: docker.io/epicmorg/app:latest
      x-squash: true

{colored('Note:', 'yellow', attrs=['bold'])}
  Buildah is invoked as a native host binary; make sure it is installed and on PATH.
"""
    print(help_text)


def show_version():
    """Print the script banner with script, Python and buildah versions."""
    print(ASCII_ART)
    print(f"Buildah Wrapper {SCRIPT_VERSION}, Python: {sys.version}")
    print(f"Buildah: {get_buildah_version()}")
