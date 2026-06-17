# Changelog
* `2.0.2.6`:
    * **Complete rewrite** from a single-file script into a packaged module (classes + `helper` layer), mirroring the `kaniko-wrapper` layout.
    * **Fixed pipe deadlock on heavy images.** `stdout` and `stderr` are now drained concurrently. The old sequential drain could fill the 64K kernel `stderr` buffer and hang the build indefinitely — the likely cause of "buildah dying" on large images.
    * Build verdict is decided **solely by the process return code** (`stdout` -> info, `stderr` -> debug); the tail of `stderr` is resurfaced at ERROR only on failure.
    * **Combinable actions:** `--build` / `--deploy` / `--clean`. `--build --deploy` builds the whole graph, then pushes, in a single run.
    * **`x-mirrors`** (compose): push a built image to additional registries during `--deploy`. A failed push to any mirror fails the service.
    * **`x-squash`** (compose): per-service single-layer control; overrides the `--squash` CLI default.
    * **`--squash` / `--no-squash`** flag via `BooleanOptionalAction`, default **off**. Per-service `x-squash` wins over it.
    * **`--build-arg` now wired** from compose `build.args` (was dropped entirely in `0.0.0.8`).
    * **Preflight check:** fail fast if the `buildah` binary is missing from `PATH` or not runnable, instead of dying mid-run on the first service.
    * `buildah build` flags moved into dataclass fields with the project's working defaults (`--isolation=oci`, `--cap-add=ALL`, `--network=host`, `--format docker`, `--layers=false`, `--no-cache`, `--disable-compression=false`, `--rm`).
    * Push uses explicit `docker://` transport for the primary image and every mirror.
    * Duplicate image names across services are rejected (a reused name would clobber the same store entry).
    * **Sequential build** (respects `FROM` chains like `light -> main -> jdk17 -> jira`); a build failure aborts before any push, so a half-built graph is never partially pushed.
    * `colorlog` logging + `pydantic-settings`; version resolved from installed package metadata.
    * Added `--verbose` / `-V` and `--log-level`.
    * Added a `pytest` suite (`tests/`).
    * `requires-python >= 3.9`.
* `0.0.0.8`: 
    * squash added as optional
* `0.0.0.7`: 
    * logs updated
    * squash disabled
* `0.0.0.6`: 
    * added `--squash`
* `0.0.0.5`: 
    * added `--network=host`
* `0.0.0.4`:
    * added `--cap-add=ALL`
* `0.0.0.3`:
    * isolation fix
* `0.0.0.2`:
    * Logic fix
* `0.0.0.1`:
    * First release
    