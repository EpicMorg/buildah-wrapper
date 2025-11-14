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

# <--- v3.0: Версия обновлена
SCRIPT_VERSION = "0.0.0.10a"

# Lock для синхронизации вывода логов в многопоточном режиме
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
    """Настройка потокобезопасного логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    # Убеждаемся, что handler потокобезопасен
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

def parse_args():
    parser = argparse.ArgumentParser(description="Buildah Wrapper", add_help=False)

    # --- Основные флаги ---
    parser.add_argument('--compose-file', default=os.getenv('COMPOSE_FILE', 'docker-compose.yml'), help='Path to docker-compose.yml file')
    parser.add_argument('--version', '-v', action='store_true', help='Show script version')
    parser.add_argument('--help', '-h', action='store_true', help='Show this help message and exit')

    # --- Флаги управления ресурсами (v3.0) ---
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel build workers (default: 4)') # <--- v3.0: Контроль параллелизма

    # --- Флаги Buildah (v3.0) ---
    parser.add_argument('--network', default='host', help='Network mode for build (default: host)') # <--- v3.0: --network
    parser.add_argument('--storage-driver', default=None, help='Storage driver (e.g., vfs). (default: system default, likely "overlay")') # <--- v3.0: --storage-driver
    parser.add_argument('--format', default='docker', help='Format of the built image (default: docker)') # <--- v3.0: --format
    parser.add_argument('--isolation', default='oci', help='Isolation mode (default: oci)') # <--- v3.0: --isolation
    parser.add_argument('--cap-add', default='ALL', help='Capabilities to add (default: ALL)') # <--- v3.0: --cap-add
    parser.add_argument('--disable-compression', default='false', help='Disable compression (default: false)') # <--- v3.0: --disable-compression
    parser.add_argument('--layers', default='false', help='Use layers (default: false)') # <--- v3.0: --layers

    # --- Флаги-переключатели Buildah (v3.0) ---
    parser.add_argument('--no-cache', action='store_true', help='Do not use cache when building') # <--- v3.0: --no-cache
    parser.add_argument('--no-rm', action='store_true', help='Do not remove intermediate containers after build') # <--- v3.0: --no-rm (инверсия --rm)
    parser.add_argument('--squash', action='store_true', help='Squash newly built layers into a single new layer')

    # --- Команды ---
    # Мы оставляем старые флаги (--build, --deploy) для обратной совместимости,
    # но основным делаем subparser
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    build_parser = subparsers.add_parser('build', help='Build images using Buildah')
    # <--- v3.0: Добавляем сюда флаги, специфичные для 'build', чтобы они работали как 'buildah-wrapper build --squash'
    build_parser.add_argument('--squash', action='store_true', help='Squash newly built layers into a single new layer')
    build_parser.add_argument('--no-cache', action='store_true', help='Do not use cache when building')

    deploy_parser = subparsers.add_parser('deploy', help='Deploy images using Buildah')
    clean_parser = subparsers.add_parser('clean', help='Clean all Buildah containers and images')

    # <--- v3.0: Старые флаги для совместимости
    parser.add_argument('--build', '-b', action='store_true', help='Build images (legacy)')
    parser.add_argument('--deploy', '-d', action='store_true', help='Deploy images (legacy)')
    parser.add_argument('--clean', action='store_true', help='Clean all (legacy)')

    return parser.parse_args()

def load_compose_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# <--- v3.0: build_with_buildah (НАСТОЯЩИЙ ФИКС)
def build_with_buildah(
    service_name: str,
    build_context: str,
    dockerfile: str,
    image_name: str,
    build_args: Dict[str, str],
    cli_args: argparse.Namespace
):
    """
    Собирает один сервис с помощью Buildah, используя настройки из cli_args.
    """

    # --- Собираем команду Buildah ---
    # Используем '=' для всех флагов, кроме тех, что этого не любят (format, f, t)
    # Это самый безопасный способ, как в v0.0.0.8
    buildah_command = [
        'buildah', 'build',
        f"--isolation={cli_args.isolation}",               # <--- ИСПРАВЛЕНО
        f"--cap-add={cli_args.cap_add}",                  # <--- ИСПРАВЛЕНО
        f"--network={cli_args.network}",                  # <--- ИСПРАВЛЕНО
        f"--disable-compression={cli_args.disable_compression}",
        '--format', cli_args.format,                     # <-- Этот флаг Ок без '='
        f"--layers={cli_args.layers}",
    ]

    if cli_args.storage_driver:
        buildah_command.extend([f"--storage-driver={cli_args.storage_driver}"]) # <--- ИСПРАВЛЕНО

    # --- Флаги-переключатели ---
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

    # --- Финальные аргументы ---
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
        # Стримим output с синхронизацией
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

# <--- v3.0: Вспомогательная функция для push
def _run_buildah_push(image_name: str) -> bool:
    """Запускает 'buildah push' для одного образа."""
    buildah_command = ['buildah', 'push', image_name]

    with _log_lock:
        logging.info(f"Deploying: {' '.join(buildah_command)}")

    process = subprocess.Popen(buildah_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        # Стримим output с синхронизацией
        for line in process.stdout:
            with _log_lock:
                logging.info(f"[{image_name}] {line.strip()}")

        process.wait()

        with _log_lock:
            if process.returncode == 0:
                logging.info(f"Successfully deployed: {image_name}")
                return True
            else:
                for line in process.stderr:
                    logging.error(f"[{image_name}] {line.strip()}")
                logging.error(f"Error deploying: {image_name}")
                return False
    except KeyboardInterrupt:
        process.terminate()
        process.wait(timeout=5)
        raise

# <--- v3.0: deploy_with_buildah переписан для x-mirrors
def deploy_with_buildah(primary_image: str, mirrors: List[str]):
    """
    Пушит основной образ и все его зеркала.
    """
    with _log_lock:
        logging.info(f"--- Deploying {primary_image} and its {len(mirrors)} mirrors ---")

    # 1. Пушим основной образ
    if not _run_buildah_push(primary_image):
        # Если не удалось запушить основной, нет смысла пушить зеркала
        raise Exception(f"Failed to deploy primary image {primary_image}")

    # 2. Пушим зеркала
    if mirrors:
        with _log_lock:
            logging.info(f"Pushing mirrors for {primary_image}...")
        failed_mirrors = 0
        for mirror in mirrors:
            if not _run_buildah_push(mirror):
                failed_mirrors += 1

        with _log_lock:
            if failed_mirrors > 0:
                logging.warning(f"Failed to push {failed_mirrors} mirrors for {primary_image}")
                # Мы не кидаем Exception, т.к. основной образ запушился
            else:
                logging.info(f"Successfully pushed all {len(mirrors)} mirrors.")

    with _log_lock:
        logging.info(f"--- Finished deploying {primary_image} ---")


def clean_buildah():
    # ... (код clean_buildah остается без изменений) ...
    # Cleaup  containers
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


def show_help():
    print(ASCII_ART)
    print(f"Buildah Wrapper v{SCRIPT_VERSION}\n")
    # <--- v3.0: Используем встроенный help от argparse, он лучше
    print("Используйте --help, -h для просмотра всех опций и команд.")


def show_version():
    buildah_version = get_buildah_version()
    print(ASCII_ART)
    print(f"Buildah Wrapper {SCRIPT_VERSION}, Python: {sys.version}")
    print(f"Buildah: {buildah_version}")

def main():
    setup_logging()

    args = parse_args()

    if args.help:
        # <--- v3.0: Печатаем ASCII и ПОЛНЫЙ help от argparse
        print(ASCII_ART)
        argparse.ArgumentParser(description=f"Buildah Wrapper v{SCRIPT_VERSION}").print_help()
        return

    if args.version:
        show_version()
        return

    # <--- v3.0: Определение команды с учетом старых флагов
    command = args.command
    if not command:
        if args.build:
            command = 'build'
        elif args.deploy:
            command = 'deploy'
        elif args.clean:
            command = 'clean'
        else:
            # Если ничего не указано, показываем версию
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

    # ... (проверка на дубликаты образов остается без изменений) ...
    for service_name, service_data in services.items():
        if not service_data: # <--- v3.0: Проверка на 'service: null'
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
        # <--- v3.0: Используем args.workers
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []

            if command == 'build':
                logging.info(f"Starting build with max {args.workers} workers...")
                for service_name, service_data in services.items():
                    if not service_data: continue # <--- v3.0: Пропуск null-сервисов

                    build_data = service_data.get('build', {})
                    if not build_data: # <--- v3.0: Пропуск сервисов без 'build'
                        logging.warning(f"No 'build' section for service {service_name}, skipping.")
                        continue

                    build_context = build_data.get('context', '.')
                    dockerfile = build_data.get('dockerfile', 'Dockerfile')
                    image_name = service_data.get('image')

                    # <--- v3.0: Парсим build-args
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
                        build_args, # <--- v3.0: Передаем build-args
                        args        # <--- v3.0: Передаем все cli_args
                    ))

            elif command == 'deploy':
                logging.info(f"Starting deploy with max {args.workers} workers...")
                for service_name, service_data in services.items():
                    if not service_data: continue # <--- v3.0: Пропуск null-сервисов

                    image_name = service_data.get('image')
                    if not image_name:
                        logging.warning(f"No image specified for service {service_name}, skipping.")
                        continue

                    # <--- v3.0: Парсим x-mirrors
                    mirrors = service_data.get('x-mirrors', [])

                    futures.append(executor.submit(
                        deploy_with_buildah,
                        image_name, # <--- v3.0: primary_image
                        mirrors     # <--- v3.0: mirrors
                    ))

            # Ждем завершения всех потоков
            for future in as_completed(futures):
                try:
                    future.result() # <--- v3.0: Выкинет Exception, если поток упал
                except Exception as exc:
                    logging.error(f"A worker failed: {exc}")
                    # (можно добавить логику для остановки остальных потоков, но пока усложнять не будем)

    except KeyboardInterrupt:
        with _log_lock:
            logging.warning("Operation interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logging.error(f"Operation failed: {exc}")
        sys.exit(1)

if __name__ == '__main__':
    main()
