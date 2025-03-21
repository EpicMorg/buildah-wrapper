# [![Activity](https://img.shields.io/github/commit-activity/m/EpicMorg/buildah-wrapper?label=commits&style=flat-square)](https://github.com/EpicMorg/buildah-wrapper/commits) [![GitHub issues](https://img.shields.io/github/issues/EpicMorg/buildah-wrapper.svg?style=popout-square)](https://github.com/EpicMorg/buildah-wrapper/issues) [![GitHub forks](https://img.shields.io/github/forks/EpicMorg/buildah-wrapper.svg?style=popout-square)](https://github.com/EpicMorg/buildah-wrapper/network) [![GitHub stars](https://img.shields.io/github/stars/EpicMorg/buildah-wrapper.svg?style=popout-square)](https://github.com/EpicMorg/buildah-wrapper/stargazers)  [![Size](https://img.shields.io/github/repo-size/EpicMorg/buildah-wrapper?label=size&style=flat-square)](https://github.com/EpicMorg/buildah-wrapper/archive/master.zip) [![Release](https://img.shields.io/github/v/release/EpicMorg/buildah-wrapper?style=flat-square)](https://github.com/EpicMorg/buildah-wrapper/releases) [![GitHub license](https://img.shields.io/github/license/EpicMorg/buildah-wrapper.svg?style=popout-square)](LICENSE.md) [![Changelog](https://img.shields.io/badge/Changelog-yellow.svg?style=popout-square)](CHANGELOG.md) [![PyPI - Downloads](https://img.shields.io/pypi/dm/buildah-wrapper?style=flat-square)](https://pypi.org/project/buildah-wrapper/)

## Description
Python wrapper for run kaniko from shell with parameters from `docker-compose.yml` file.

## Motivation
1. You have Docker project thar contains:
1.1 `docker-compose.yml` - as build manifest
1.2 One or more `Dockerfile`s in project
2. You want to automate builds with `kaniko` build system.
3. `kaniko` dont support `docker-compose.yml` builds.

## How to
```
pip install buildah-wrapper
cd <...>/directory/contains/docker/and/docker-compose-file/
buildah-wrapper
```

### Arguments (examples)
* `--compose-file` - Path to docker-compose.yml file
* `--version`, `-v` - Show script version
* `--help`, `-h` - Show this help message and exit

## Supported features (example):

1. Single project in `docker-compose.yml`
```
services:
  app:
    image: "EpicMorg/buildah-wrapper:image"
    build:
      context: .
      dockerfile: ./Dockerfile
```

2. Multiproject in `docker-compose.yml`

```
services:
  app:
    image: "EpicMorg/buildah-wrapper:image-jdk11"
    build:
      context: .
  app-develop:
    image: "EpicMorg/buildah-wrapper:image-develop-jdk11"
    build:
      context: .
      dockerfile: ./Dockerfile.develop
  app-develop-17:
    image: "epicmorg/astralinux:image-develop-jdk17"
    build:
      context: .
      dockerfile: ./Dockerfile.develop-17
```
