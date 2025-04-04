import os
import argparse
import yaml
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import sys

# Script version
SCRIPT_VERSION = "0.0.0.7"

# ASCII art for Buildah Wrapper
ASCII_ART = r"""
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

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_buildah_version():
    """Get version of Buildah."""
    try:
        result = subprocess.run(['buildah', '-v'], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get Buildah version: {e}")
        return "Unknown"

def parse_args():
    parser = argparse.ArgumentParser(description="Buildah Wrapper", add_help=False)
    parser.add_argument('--compose-file', default=os.getenv('COMPOSE_FILE', 'docker-compose.yml'), help='Path to docker-compose.yml file')
    parser.add_argument('--version', '-v', action='store_true', help='Show script version')
    parser.add_argument('--help', '-h', action='store_true', help='Show this help message and exit')
    
    # Add --build, -b, --deploy, -d, --clean
    parser.add_argument('--build', '-b', action='store_true', help='Build images using Buildah')
    parser.add_argument('--deploy', '-d', action='store_true', help='Deploy images using Buildah')
    parser.add_argument('--clean', action='store_true', help='Clean all Buildah containers and images')
    
    # Subs build, deploy и clean
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    build_parser = subparsers.add_parser('build', help='Build images using Buildah')
    deploy_parser = subparsers.add_parser('deploy', help='Deploy images using Buildah')
    clean_parser = subparsers.add_parser('clean', help='Clean all Buildah containers and images')

    return parser.parse_args()

def load_compose_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def build_with_buildah(service_name, build_context, dockerfile, image_name):
    buildah_command = [
        'buildah', 'build',
        '--isolation=oci',
        '--cap-add=ALL',
        '--network=host',
        '--disable-compression=false',
        '--format', 'docker', # i am not want use oci format because it not support heathcheks
        '--no-cache',
        '--rm',
        '--layers=false',
#        '--squash',
        '-f', f'{build_context}/{dockerfile}',
        '-t', image_name,
        build_context
    ]

    logging.info(f"Building {service_name} with Buildah:")
    logging.info(f"{' '.join(buildah_command)}")
    
    process = subprocess.Popen(buildah_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Stream output in real-time
    for line in process.stdout:
        logging.info(line.strip())
    
    process.wait()
    
    if process.returncode == 0:
        logging.info(f"Successfully built {service_name}")
    else:
        for line in process.stderr:
            logging.error(line.strip())
        logging.error(f"Error building of {service_name}")
        raise Exception(f"Failed to build {service_name}")

def deploy_with_buildah(image_name):
    buildah_command = [
        'buildah', 'push',
        image_name
    ]

    logging.info(f"Deploying service with Buildah:")
    logging.info(f"{' '.join(buildah_command)}")
    
    process = subprocess.Popen(buildah_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Stream output in real-time
    for line in process.stdout:
        logging.info(line.strip())
    
    process.wait()
    
    if process.returncode == 0:
        logging.info(f"Successfully deployed:")
        logging.info(f"{' '.join(image_name)}")
    else:
        for line in process.stderr:
            logging.error(line.strip())
        logging.error(f"Error deploying of:")
        logging.error(f"{' '.join(image_name)}")
        raise Exception(f"Failed to deploy {image_name}")

def clean_buildah():
    # Cleaup  containers
    rm_command = ['buildah', 'rm', '--all']
    logging.info(f"Cleaning Buildah containers:")
    logging.info(f"{' '.join(rm_command)}")
    
    rm_process = subprocess.Popen(rm_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in rm_process.stdout:
        logging.info(line.strip())
    rm_process.wait()
    
    if rm_process.returncode != 0:
        for line in rm_process.stderr:
            logging.error(line.strip())
        logging.error("Error cleaning Buildah containers")
        raise Exception("Failed to clean Buildah containers")
    
    # Cleanup images
    rmi_command = ['buildah', 'rmi', '--all']
    logging.info(f"Cleaning Buildah images:")
    logging.info(f"{' '.join(rmi_command)}")
    
    rmi_process = subprocess.Popen(rmi_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in rmi_process.stdout:
        logging.info(line.strip())
    rmi_process.wait()
    
    if rmi_process.returncode != 0:
        for line in rmi_process.stderr:
            logging.error(line.strip())
        logging.error("Error cleaning Buildah images")
        raise Exception("Failed to clean Buildah images")
    
    logging.info("Successfully cleaned all Buildah containers and images")

def show_help():
    print(ASCII_ART)
    print("Buildah Wrapper\n")
    print("Arguments:")
    print("--compose-file        Path to docker-compose.yml file")
    print("--version, -v         Show script version")
    print("--help, -h            Show this help message and exit")
    print("\nCommands:")
    print("--build, -b           Build images using Buildah")
    print("--deploy, -d          Deploy images using Buildah")
    print("--clean               Clean all Buildah containers and images")

def show_version():
    buildah_version = get_buildah_version()
    print(ASCII_ART)
    print(f"Buildah Wrapper {SCRIPT_VERSION}, Python: {sys.version}")
    print(f"Buildah: {buildah_version}")

def main():
    setup_logging()
    
    args = parse_args()

    # Show help and exit if --help is provided
    if args.help:
        show_help()
        return
    
    # Show version and exit if --version or no relevant arguments are provided
    if args.version or not (args.build or args.deploy or args.clean or args.command):
        show_version()
        return
    
    
    if args.clean or args.command == 'clean':
        try:
            clean_buildah()
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
    
    for service_name, service_data in services.items():
        image_name = service_data.get('image')
        
        if not image_name:
            logging.warning(f"No image specified for service {service_name}")
            continue
        
        image_names[image_name] += 1
    
    for image_name, count in image_names.items():
        if count > 1:
            logging.error(f"Error: Image name {image_name} is used {count} times.")
            return
    
    try:
        
        command = args.command
        if args.build:
            command = 'build'
        elif args.deploy:
            command = 'deploy'
        
        if command == 'build':
            with ThreadPoolExecutor() as executor:
                futures = []
                for service_name, service_data in services.items():
                    build_data = service_data.get('build', {})
                    build_context = build_data.get('context', '.')
                    dockerfile = build_data.get('dockerfile', 'Dockerfile')
                    image_name = service_data.get('image')
                    
                    if not image_name:
                        logging.warning(f"No image specified for service {service_name}")
                        continue
                    
                    futures.append(executor.submit(build_with_buildah, service_name, build_context, dockerfile, image_name))
                
                for future in as_completed(futures):
                    future.result()
        
        elif command == 'deploy':
            with ThreadPoolExecutor() as executor:
                futures = []
                for service_name, service_data in services.items():
                    image_name = service_data.get('image')
                    
                    if not image_name:
                        logging.warning(f"No image specified for service {service_name}")
                        continue
                    
                    futures.append(executor.submit(deploy_with_buildah, image_name))
                
                for future in as_completed(futures):
                    future.result()
    
    except Exception as exc:
        logging.error(f"Operation failed: {exc}")
        sys.exit(1)

if __name__ == '__main__':
    main()
