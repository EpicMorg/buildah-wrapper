# [![Activity](https://img.shields.io/github/commit-activity/m/EpicMorg/buildah-wrapper?label=commits&style=flat-square)](https://github.com/EpicMorg/buildah-wrapper/commits) [![GitHub issues](https://img.shields.io/github/issues/EpicMorg/buildah-wrapper.svg?style=popout-square)](https://github.com/EpicMorg/buildah-wrapper/issues) [![GitHub forks](https://img.shields.io/github/forks/EpicMorg/buildah-wrapper.svg?style=popout-square)](https://github.com/EpicMorg/buildah-wrapper/network) [![GitHub stars](https://img.shields.io/github/stars/EpicMorg/buildah-wrapper.svg?style=popout-square)](https://github.com/EpicMorg/buildah-wrapper/stargazers)  [![Size](https://img.shields.io/github/repo-size/EpicMorg/buildah-wrapper?label=size&style=flat-square)](https://github.com/EpicMorg/buildah-wrapper/archive/master.zip) [![Release](https://img.shields.io/github/v/release/EpicMorg/buildah-wrapper?style=flat-square)](https://github.com/EpicMorg/buildah-wrapper/releases) [![GitHub license](https://img.shields.io/github/license/EpicMorg/buildah-wrapper.svg?style=popout-square)](LICENSE.md) [![Changelog](https://img.shields.io/badge/Changelog-yellow.svg?style=popout-square)](CHANGELOG.md) [![PyPI - Downloads](https://img.shields.io/pypi/dm/buildah-wrapper?style=flat-square)](https://pypi.org/project/buildah-wrapper/)

## Description
Python wrapper to run [Buildah](https://buildah.io/) from the shell, driven by the build manifest in a `docker-compose.yml` file.

## Motivation
1. You have a Docker project that contains:
    1. `docker-compose.yml` — as the build manifest
    2. one or more `Dockerfile`s in the project
2. You want to automate builds with the `buildah` build system.
3. `buildah` does not natively consume `docker-compose.yml` for builds.

## Requirements
* Linux with the `buildah` binary on `PATH` (invoked as a native host binary — there is no executor image to pull).
* Python `>= 3.9`.

## How to
```
pip install buildah-wrapper
cd <...>/directory/containing/docker-and-docker-compose-file/
buildah-wrapper --build            # build into the local store
buildah-wrapper --build --deploy   # build, then push to the registry and mirrors
buildah-wrapper --deploy           # push already-built images
buildah-wrapper --clean            # remove all buildah containers and images
```
Running with no action prints the help screen.

### Arguments
Actions are combinable (e.g. `--build --deploy`):
* `--build`, `-b` — build images into the local store
* `--deploy`, `-d` — push built images to the registry and all `x-mirrors`
* `--clean` — remove all buildah containers and images

Options:
* `--compose-file FILE` — path to the `docker-compose.yml` file (default: `docker-compose.yml`)
* `--squash` / `--no-squash` — single-layer output for all services (default: **off**; per-service `x-squash` overrides it)
* `--verbose`, `-V` — verbose output (shortcut for `--log-level DEBUG`)
* `--log-level LEVEL` — override log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
* `--version`, `-v` — show script, Python and buildah versions
* `--help`, `-h` — show the help message and exit

## Supported features

### 1. Single project in `docker-compose.yml`
```yaml
services:
  app:
    image: "epicmorg/buildah-wrapper:image"
    build:
      context: .
      dockerfile: ./Dockerfile
```

### 2. Multi-project in `docker-compose.yml`
```yaml
services:
  app:
    image: "epicmorg/buildah-wrapper:image-jdk11"
    build:
      context: .
  app-develop:
    image: "epicmorg/buildah-wrapper:image-develop-jdk11"
    build:
      context: .
      dockerfile: ./Dockerfile.develop
  app-develop-17:
    image: "epicmorg/astralinux:image-develop-jdk17"
    build:
      context: .
      dockerfile: ./Dockerfile.develop-17
```

### 3. Mirrors — `x-mirrors`
Add an `x-mirrors` list to a service to push the built image to additional
registries during `--deploy`. A failed push to any mirror fails the service.
```yaml
services:
  app:
    image: docker.io/epicmorg/app:latest
    build:
      context: .
    x-mirrors:
      - quay.io/epicmorg/app:latest
      - ghcr.io/epicmorg/app:latest
```

### 4. Squash — `x-squash`
Add an `x-squash` field to a service to control single-layer output per service.
Default is `false`. A per-service `x-squash` overrides the `--squash` / `--no-squash`
CLI default for that service.
```yaml
services:
  app:
    image: docker.io/epicmorg/app:latest
    build:
      context: .
    x-squash: true
```

## Notes
* `buildah build` runs with `--format docker` (OCI format does not carry `HEALTHCHECK`).
* Builds run sequentially, so `FROM` chains (`base -> runtime -> app`) resolve in order.
* Pushes use the default registry auth (`~/.config/containers/auth.json` or `REGISTRY_AUTH_FILE`).
