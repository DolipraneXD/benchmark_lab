# Design: every benchmark run in its own container

## Intent

BenchLab will move to a server on the lab's private network. Until then it is developed
on a laptop. The user wants every benchmark, including the small API-only ones, to run
inside Docker, and the project to be deployable the usual DevOps way. Containers give:
- the same environment on the laptop and the server
- a boundary around each run, so a crash or a runaway harness can't take down the panel
  or other runs
- CPU and memory limits per run

Security for the shared server (what lab members may do without a login, spending caps,
who started what) is a **later step**. This design only has to leave room for it.

## Goals

- Every run started from the panel runs in its own container, created for that run and
  removed when it ends.
- A run in a container is the **same `benchlab run …` command** as today. Same folders,
  same `result.json`, same resume and budget behaviour. The panel shows it exactly as now.
- Images are built from the repo with pinned versions, so the laptop and the server run
  the same thing.
- Each run gets CPU and memory limits, set per benchmark family in the config.
- Stop still saves progress (Ctrl-C semantics), and Resume still continues.
- `docker compose up` starts the panel on the laptop. The same compose file, plus an
  override, is the start of the server deployment.
- All 14 benchmarks that are ready today still work: they pass a smoke run through
  containers. GAIA and BrowseComp also work once their keys are added.

## Non-goals (for now)

- Server security: login-free access rules, spending caps, audit of who ran what.
- TLS, a reverse proxy, backups, CI/CD pipelines (the deployment step after this one).
- GPU scheduling for self-hosted models or retrieval. CPU only, as today.
- Kubernetes.

## Anti-goals (a failure even if it works)

- A containerized run that behaves differently from the terminal run (other params,
  other folders, other resume behaviour).
- API keys baked into an image, written into a run folder, or visible in `docker inspect`
  of an image layer.
- The panel needing unrestricted access to the host's Docker. It gets only what starting
  and stopping runs needs, through a socket proxy.
- Dropping the laptop workflow: `benchlab ui` and `benchlab run` without Docker keep working
  for development.

## Constraints

- **Laptop:** 8 CPUs, 16 GB RAM, Docker 29.8, user in the `docker` group.
- **Harness installs are large:** 9.6 GB today. Retrieval (PyTorch, sentence-transformers)
  is 5.6 GB, τ²-bench 1.1 GB, LiveCodeBench 1.2 GB.
- **Four benchmarks start their own containers:**
  - Terminal-Bench and SWE-bench, through Harbor, which uses Docker Compose per task.
  - GAIA and BrowseComp: the Inspect sandbox.
  - Aider, with `docker run`.
- **Aider is the only one whose inner container calls the model.** Today it uses
  `--network=host` to reach BenchLab's LiteLLM proxy on the host. The others call the model
  from the harness process, and their containers only run code.
- **Aider mounts host paths into its container** (`-v <path>:/aider`). A container started
  from inside another container is created by the host's Docker, so mount paths are host
  paths.

## Approaches considered

- **A. One BenchLab image holding every harness.** Runs happen inside that one container,
  using the host Docker socket for the inner containers. Simplest. But every run shares one
  container's CPU and memory, updating one harness rebuilds everything, and the whole app
  has full Docker access.
- **B. A panel container, plus one container per run, from per-family images.** *Chosen by
  the user.* Isolation and limits per run, harnesses updated per family, a clear path to the
  server. Costs more images and more work up front.
- **C. Docker-in-Docker (Sysbox).** No host socket, but it needs a special runtime on the
  host, is slower, and adds complexity without solving more than B. Rejected.

## How it works (decided)

### Images (all built from `docker/` in the repo, pinned)

| Image | Contains | Used by |
|---|---|---|
| `benchlab-core` | Python 3.12, uv, BenchLab, LiteLLM, the Inspect, τ²-bench and Harbor harnesses, the Docker CLI with the compose plugin | the panel, and runs of app tasks, GPQA, BFCL, TTFT, τ²-bench, GAIA, BrowseComp, Terminal-Bench, SWE-bench, the red-team import |
| `benchlab-code` | `benchlab-core` + the LiveCodeBench and AutomationBench harnesses | runs of LiveCodeBench, AutomationBench, Aider |
| `benchlab-retrieval` | `benchlab-core` + the retrieval harness (PyTorch CPU build, sentence-transformers, mteb) | MTEB, BEIR, own retrieval |
| `benchlab-serving` | `benchlab-core` + guidellm | the serving benchmark |
| `benchlab-aider-task` | Aider's own benchmark image, built from its pinned commit | the containers Aider starts for each exercise |

- **Harnesses are installed while the image builds**, into `/opt/benchlab/harnesses`, by
  running each adapter's own `prepare()`. One install path, already pinned and tested.
  `BENCHLAB_HARNESS_DIR` points there.
- **What stays outside the image:** the workspace (runs, results) and the configs, which
  are mounted.
- **Which image a benchmark uses** is a new `image` field on each registry entry. The
  default is `benchlab-core`.

### Runs as containers

- **A new executor setting** (`settings.executor: local | docker`, default `local`) decides
  how the panel starts a job.
  - **`local`** is today's detached subprocess.
  - **`docker`** runs the same argv in a new container:
    `docker run --rm --name benchlab-job-<id> --network benchlab --cpus <c> --memory <m>
    -v <workspace>:<workspace> -v <configs>:/app/<configs>:ro --env <only the needed keys>
    <image> benchlab run -c … -m … -b …`
- **A job running more than one benchmark family** is split into one container per family,
  run one after another. The job record lists the containers.
- **The workspace is mounted at the same absolute path inside every container as on the
  host** (path parity). Any mount a harness passes to Docker (Aider's `-v`) is then a valid
  host path.
- **Logs:** the container's stdout and stderr go to the job's `.log` through the existing
  launcher, which waits on `docker run`. The exit code goes to the `.exit` file as today, so
  status, progress and the "skipped" reasons work unchanged.
- **Stop** sends `docker kill --signal=SIGINT benchlab-job-<id>`. The CLI inside catches
  Ctrl-C, saves state and exits 130, as today. Resume starts the same job again.
- **Keys:** the panel passes `--env NAME` (the value is taken from its own environment)
  only for the key variables the job's models and helper models use. The run never sees
  the other keys, and they never appear on a command line.
- **Limits:** `settings.container_limits` per family, e.g.
  `core: {cpus: 2, memory: 4g}`, `code: {cpus: 4, memory: 8g}`,
  `retrieval: {cpus: 4, memory: 8g}`. Containers that Harbor and Inspect start get their
  limits from those harnesses (Harbor already sets per-task resources).

### Networking

- **A user-defined Docker network `benchlab`** holds the panel, the socket proxy and every
  run container.
- **Inside a run container, BenchLab's LiteLLM proxy listens on `0.0.0.0`** at its random
  port, and its address is the container's name (`benchlab-job-<id>`).
- **Aider's exercise containers** join the `benchlab` network instead of the host network,
  and reach the proxy at `http://benchlab-job-<id>:<port>/v1`. With the `local` executor,
  nothing changes.

### Access to Docker

- **The panel** gets Docker access only through a socket proxy (`docker-socket-proxy`),
  allowing just what is needed to create, start, inspect, signal and remove containers.
- **Run containers of the four Docker-using benchmarks** also need Docker: Harbor uses
  compose (networks, volumes, image pulls, exec), Inspect's sandbox needs the same, and
  Aider needs `docker run`. They reach Docker through the same proxy, with
  those extra permissions enabled. **This is the one place a run can affect the host's
  Docker.** It is documented, and it is limited to those benchmarks: the other run
  containers get no Docker access at all.

### Compose and dev workflow

- **`docker/compose.yaml`** has three services:
  - `panel`: `benchlab ui` bound to `0.0.0.0` inside the container, published on
    `127.0.0.1:8765` on the laptop.
  - `docker-proxy`.
  - a one-shot `build` profile that builds all images.
- **`.env` is passed to the panel** with `env_file`. On the server this becomes Docker
  secrets, in the deployment step.
- **The panel in a container needs two changes:** a `--host` option, and a token passed in
  or printed to the container logs. The one-time-link security model stays.
- **`benchlab ui` and `benchlab run` on the host keep working** with the `local` executor,
  for development and debugging.

### Verification

- **Unit tests** cover:
  - the docker argv the executor builds: limits, path parity, network, only the needed key
    names and never values, the image chosen per family
  - stop sending SIGINT
  - splitting a multi-family job
- **Integration test:** run a smoke run of `ttft` and `apptasks` through the `docker`
  executor against the mock server on the `benchlab` network. It is skipped when Docker or
  the images are missing.
- **A manual smoke run of every benchmark** through containers on the laptop with minimax
  (a few cents each), producing a table of what passes. Anything that fails gets fixed or
  reported with its reason. This answers "benchmarks still fail too much" with evidence.

## Left to the builder

- Dockerfile details: base OS, multi-stage layout, layer caching.
- Exact socket-proxy permission flags, per access level.
- How the panel learns the host-side workspace path for path parity: an env var set by
  compose is the expected answer.
- The job record format for multi-container jobs.
- How images are tagged: with BenchLab's version plus the git commit.

## Calls I made (please check)

1. **Harnesses are baked into images** by running each adapter's `prepare()` at build time,
   rather than installed on first use into the workspace as today. This is more
   reproducible and matches the server, but image builds take longer: retrieval alone is
   5.6 GB.
2. **The images are split into four families** (core, code, retrieval, serving), with core
   used by the panel too. The heavy retrieval stack isn't in every run.
3. **The workspace is mounted at the same path inside containers as on the host**, so
   containers started from inside a run (Aider) get valid mounts.
4. **The four benchmarks that start their own containers get Docker access through the
   socket proxy.** The others get none. This is the known risk of approach B, and it is
   limited and documented.
5. **The `local` executor stays the default** for now. Docker is switched on by config,
   and compose sets it.
6. **Default limits:** core 2 CPUs / 4 GB, code and retrieval 4 CPUs / 8 GB. Change them in
   the config.
