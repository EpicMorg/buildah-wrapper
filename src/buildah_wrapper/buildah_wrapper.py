#!/usr/bin/env python3

import os
import argparse
import yaml
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import sys
import threading
from typing import List, Dict, Any, Optional

# Updated script version
SCRIPT_VERSION = "1.0.0.0"

# Lock for synchronizing log output in threaded mode
_log_lock = threading.Lock()

# ASCII art for Buildah Wrapper
ASCII_ART = r"""
+=========================================================================+
 /$$$$$$$$         /$$         /$$      /$$
| $$_____/        |__/        | $$$    /$$$
| $$       /$$$$$$ /$$ /$$$$$$| $$$$  /$$$$ /$$$$$$  /$$$$$$  /$$$$$$
| $$$$$   /$$__  $| $$/$$_____| $$ $$/$$ $$/$$__  $$/$$__  $$/$$__  $$
| $$__/  | $$  \ $| $| $$     | $$  $$$| $| $$  \ $| $$  \__| $$  \ $$
| $$     | $$  | $| $| $$     | $$\  $ | $| $$  | $| $$     | $$
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

def setup_logging():
    """Setup thread-safe logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    # Ensure the handler is thread-safe
    for handler in logging.root.handlers:
        handler.setLevel(logging.INFO)

def get_buildah_version():
    """Get version of Buildah."""
    try:
        result = subprocess.run(['buildah', '-v'], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get Buildah version: {e}")
        return "Unknown"

def create_parser() -> argparse.ArgumentParser: # <--- RENAMED (from previous fix)
    """Creates and configures the argument parser."""
    parser = argparse.ArgumentParser(description="Buildah Wrapper", add_help=False)

    # --- Core Flags ---
    parser.add_argument('--compose-file', default=os.getenv('COMPOSE_FILE', 'docker-compose.yml'), help='Path to docker-compose.yml file')
    parser.add_argument('--version', '-v', action='store_true', help='Show script version')
    parser.add_argument('--help', '-h', action='store_true', help='Show this help message and exit')

    # --- Resource Management Flags ---
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel build workers (default: 4)')

    # --- Buildah Flags ---
    parser.add_argument('--network', default='host', help='Network mode for build (default: host)')
    parser.add_argument('--storage-driver', default=None, help='Storage driver (e.g., vfs). (default: system default, likely "overlay")')
    parser.add_argument('--format', default='docker', help='Format of the built image (default: docker)')
    parser.add_argument('--isolation', default='oci', help='Isolation mode (default: oci)')
    parser.add_argument('--cap-add', default='ALL', help='Capabilities to add (default: ALL)')
    parser.add_argument('--disable-compression', default='false', help='Disable compression (default: false)')
    parser.add_argument('--layers', default='false', help='Use layers (default: false)')

    # --- Buildah Toggle Flags ---
    parser.add_argument('--no-cache', action='store_true', help='Do not use cache when building')
    parser.add_argument('--no-rm', action='store_true', help='Do not remove intermediate containers after build')
    parser.add_argument('--squash', action='store_true', help='Squash newly built layers into a single new layer')

    # --- Commands ---
    # We keep the old flags for backward compatibility,
    # but the subparser is now the primary method
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    build_parser = subparsers.add_parser('build', help='Build images using Buildah')
    # Add build-specific flags here so 'buildah-wrapper build --squash' works
    build_parser.add_argument('--squash', action='store_true', help='Squash newly built layers into a single new layer')
    build_parser.add_argument('--no-cache', action='store_true', help='Do not use cache when building')

    deploy_parser = subparsers.add_parser('deploy', help='Deploy images using Buildah')
    clean_parser = subparsers.add_parser('clean', help='Clean all Buildah containers and images')

    # Legacy flags for compatibility
    parser.add_argument('--build', '-b', action='store_true', help='Build images (legacy)')
    parser.add_argument('--deploy', '-d', action='store_true', help='Deploy images (legacy)')
    parser.add_argument('--clean', action='store_true', help='Clean all (legacy)')

    return parser

def load_compose_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def build_with_buildah(
    service_name: str,
    build_context: str,
    dockerfile: str,
    image_name: str,
    build_args: Dict[str, str],
    cli_args: argparse.Namespace
):
    """
    Builds a single service using Buildah, applying settings from cli_args.
    """

    # --- Assemble Buildah Command ---
    # Use '=' for all flags except those that dislike it (format, f, t)
    # This is the safest method, as seen in v0.0.0.8
    buildah_command = [
        'buildah', 'build',
        f"--isolation={cli_args.isolation}",
        f"--cap-add={cli_args.cap_add}",
        f"--network={cli_args.network}",
        f"--disable-compression={cli_args.disable_compression}",
        '--format', cli_args.format, # <-- This flag is OK without '='
        f"--layers={cli_args.layers}",
    ]

    if cli_args.storage_driver:
        buildah_command.extend([f"--storage-driver={cli_args.storage_driver}"])

    # --- Toggle Flags ---
    if cli_args.no_cache or (cli_args.command == 'build' and getattr(cli_args, 'no_cache', False)):
        buildah_command.append('--no-cache')

    if not cli_args.no_rm:
        buildah_command.append('--rm')

    if cli_args.squash or (cli_args.command == 'build' and getattr(cli_args, 'squash', False)):
        buildah_command.append('--squash')

    # --- Build Args ---
    if build_args:
        for key, value in build_args.items():
            buildah_command.extend(['--build-arg', f"{key}={value}"])

    # --- Final Arguments ---
    buildah_command.extend([
        '-f', f'{build_context}/{dockerfile}',
        '-t', image_name,
        build_context
    ])

    with _log_lock:
        logging.info(f"Building {service_name} with Buildah:")
        logging.info(f"{' '.join(buildah_command)}")

    process = subprocess.Popen(buildah_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        # Stream output with synchronization
        for line in process.stdout:
            with _log_lock:
                logging.info(f"[{service_name}] {line.strip()}")

        process.wait()

        with _log_lock:
            if process.returncode == 0:
                logging.info(f"Successfully built {service_name}")
            else:
                for line in process.stderr:
                    logging.error(f"[{service_name}] {line.strip()}")
                logging.error(f"Error building of {service_name}")
        
        if process.returncode != 0:
            raise Exception(f"Failed to build {service_name}")
    except KeyboardInterrupt:
        process.terminate()
        process.wait(timeout=5)
        raise

# <--- MODIFIED: Function signature changed
def _run_buildah_push(local_image_name: str, remote_destination: str) -> bool:
    """Pushes a local image to a remote destination."""
    
    # <--- MODIFIED: Fixed push logic
    buildah_command = ['buildah', 'push', local_image_name, f"docker://{remote_destination}"]

    with _log_lock:
        logging.info(f"Deploying: {' '.join(buildah_command)}")

    process = subprocess.Popen(buildah_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        # Stream output with synchronization
        for line in process.stdout:
            with _log_lock:
                # <--- MODIFIED: Log by destination
                logging.info(f"[{remote_destination}] {line.strip()}")

        process.wait()

        with _log_lock:
            if process.returncode == 0:
                # <--- MODIFIED: Log by destination
                logging.info(f"Successfully deployed: {remote_destination}")
                return True
            else:
                for line in process.stderr:
                    # <--- MODIFIED: Log by destination
                    logging.error(f"[{remote_destination}] {line.strip()}")
                # <--- MODIFIED: Log by destination
                logging.error(f"Error deploying: {remote_destination}")
                return False
    except KeyboardInterrupt:
        process.terminate()
        process.wait(timeout=5)
        raise

# <--- MODIFIED: Updated with fixed deploy logic
def deploy_with_buildah(primary_image: str, mirrors: List[str]):
    """
    Pushes the primary image (local name) to its primary destination and all mirrors.
    """
    with _log_lock:
        logging.info(f"--- Deploying {primary_image} and its {len(mirrors)} mirrors ---")

    # 1. Push primary image (local_name == remote_destination)
    if not _run_buildah_push(primary_image, primary_image):
        # If the primary push fails, no point in pushing mirrors
        raise Exception(f"Failed to deploy primary image {primary_image}")

    # 2. Push mirrors
    if mirrors:
        with _log_lock:
            logging.info(f"Pushing mirrors for {primary_image}...")
        failed_mirrors = 0
        for mirror_destination in mirrors:
            # Push the LOCAL image (primary_image) to the REMOTE mirror (mirror_destination)
            if not _run_buildah_push(primary_image, mirror_destination):
                failed_mirrors += 1

        with _log_lock:
            if failed_mirrors > 0:
                logging.warning(f"Failed to push {failed_mirrors} mirrors for {primary_image}")
                # We don't raise an Exception, as the primary image was pushed
            else:
                logging.info(f"Successfully pushed all {len(mirrors)} mirrors.")

    with _log_lock:
        logging.info(f"--- Finished deploying {primary_image} ---")


def clean_buildah():
    # Cleanup containers
    rm_command = ['buildah', 'rm', '--all']
    with _log_lock:
        logging.info(f"Cleaning Buildah containers:")
        logging.info(f"{' '.join(rm_command)}")

    rm_process = subprocess.Popen(rm_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    try:
        for line in rm_process.stdout:
            with _log_lock:
                logging.info(line.strip())
        rm_process.wait()

        with _log_lock:
            if rm_process.returncode != 0:
                for line in rm_process.stderr:
                    logging.error(line.strip())
                logging.error("Error cleaning Buildah containers")
        
        if rm_process.returncode != 0:
            raise Exception("Failed to clean Buildah containers")
    except KeyboardInterrupt:
        rm_process.terminate()
        rm_process.wait(timeout=5)
        raise

    # Cleanup images
    rmi_command = ['buildah', 'rmi', '--all']
    with _log_lock:
        logging.info(f"Cleaning Buildah images:")
        logging.info(f"{' '.join(rmi_command)}")

    rmi_process = subprocess.Popen(rmi_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    try:
        for line in rmi_process.stdout:
            with _log_lock:
                logging.info(line.strip())
        rmi_process.wait()

        with _log_lock:
            if rmi_process.returncode != 0:
                for line in rmi_process.stderr:
                    logging.error(line.strip())
                logging.error("Error cleaning Buildah images")
        
        if rmi_process.returncode != 0:
            raise Exception("Failed to clean Buildah images")

        with _log_lock:
            logging.info("Successfully cleaned all Buildah containers and images")
    except KeyboardInterrupt:
        rmi_process.terminate()
        rmi_process.wait(timeout=5)
        raise


def show_help(parser: argparse.ArgumentParser): # <--- MODIFIED
    """Prints the custom ASCII art and the parser's help."""
    print(ASCII_ART)
    print(f"Buildah Wrapper v{SCRIPT_VERSION}\n")
    # Print help from the *actual* parser
    parser.print_help() # <--- MODIFIED


def show_version():
    buildah_version = get_buildah_version()
    print(ASCII_ART)
    print(f"Buildah Wrapper {SCRIPT_VERSION}, Python: {sys.version}")
    print(f"Buildah: {buildah_version}")

def main():
    setup_logging()

    # --- MODIFIED: Refactored for correct --help ---
    # 1. Create the parser
    parser = create_parser()

    # 2. Parse arguments
    args = parser.parse_args()

    # 3. Check for --help (and pass the parser)
    if args.help:
        show_help(parser)
        return
    # --- END MODIFICATION ---

    if args.version:
        show_version()
        return

    # Determine command, respecting legacy flags
    command = args.command
    if not command:
        if args.build:
            command = 'build'
        elif args.deploy:
            command = 'deploy'
        elif args.clean:
            command = 'clean'
        else:
            # If no command is specified, show version
            show_version()
            return

    if command == 'clean':
        try:
            clean_buildah()
        except KeyboardInterrupt:
            with _log_lock:
                logging.warning("Clean interrupted by user.")
            sys.exit(130)
        except Exception as exc:
            logging.error(f"Clean failed: {exc}")
            sys.exit(1)
        return

    compose_file = args.compose_file

    if not os.path.exists(compose_file):
        logging.error(f"{compose_file} not found")
        return

    compose_data = load_compose_file(compose_file)

    services = compose_data.get('services', {})
    image_names = defaultdict(int)

    # ... (duplicate image check remains unchanged) ...
    for service_name, service_data in services.items():
        if not service_data: # Check for 'service: null'
            logging.warning(f"Service {service_name} is empty (null) in compose file, skipping.")
            continue
        image_name = service_data.get('image')
        if not image_name:
            logging.warning(f"No image specified for service {service_name}, skipping.")
            continue
        image_names[image_name] += 1

    for image_name, count in image_names.items():
        if count > 1:
            logging.error(f"Error: Image name {image_name} is used {count} times.")
            return

    try:
        # Use args.workers
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []

            if command == 'build':
                logging.info(f"Starting build with max {args.workers} workers...")
                for service_name, service_data in services.items():
                    if not service_data: continue # Skip null services

                    build_data = service_data.get('build', {})
                    if not build_data: # Skip services without 'build'
                        logging.warning(f"No 'build' section for service {service_name}, skipping.")
                        continue

                    build_context = build_data.get('context', '.')
                    dockerfile = build_data.get('dockerfile', 'Dockerfile')
                    image_name = service_data.get('image')

                    # Parse build-args
                    build_args = build_data.get('args', {})

                    if not image_name:
                        logging.warning(f"No image specified for service {service_name}, skipping.")
                        continue

                    futures.append(executor.submit(
                        build_with_buildah,
                        service_name,
                        build_context,
                        dockerfile,
                        image_name,
                        build_args, # Pass build-args
                        args        # Pass all cli_args
                    ))

            elif command == 'deploy':
                logging.info(f"Starting deploy with max {args.workers} workers...")
                for service_name, service_data in services.items():
                    if not service_data: continue # Skip null services

                    image_name = service_data.get('image')
                    if not image_name:
                        logging.warning(f"No image specified for service {service_name}, skipping.")
                        continue

                    # Parse x-mirrors
                    mirrors = service_data.get('x-mirrors', [])

                    futures.append(executor.submit(
                        deploy_with_buildah,
                        image_name, # primary_image
                        mirrors     # mirrors
                    ))

            # Wait for all threads to complete
            for future in as_completed(futures):
                try:
                    future.result() # Will re-throw exception if a worker failed
                except Exception as exc:
                    logging.error(f"A worker failed: {exc}")
                    # (could add logic to stop other threads, but let's keep it simple for now)

    except KeyboardInterrupt:
        with _log_lock:
            logging.warning("Operation interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logging.error(f"Operation failed: {exc}")
        sys.exit(1)

if __name__ == '__main__':
    main()