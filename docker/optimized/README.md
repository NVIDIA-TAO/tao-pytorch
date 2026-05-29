# Optimized base-image build

Two-tier, content-addressable rebuild of the TAO PyTorch base image. Compiles
each heavy from-source component (ONNX Runtime, torch-scatter, WarpConvNet)
once into its own artifact image keyed by content hash, then assembles the
final base image by copying the pre-built wheels in. Re-runs that don't change
inputs become registry-tag lookups instead of multi-hour rebuilds.

Result on a no-pin-change run: ~6 manifest lookups (~1 s each) plus two Tier‑2
pip-install builds (~5 min each), versus ~2 h for the legacy monolithic build.

## Layout

```
docker/optimized/
├── Dockerfile.optimized            # Tier 2: thin assembly Dockerfile
├── artifacts/                      # Tier 1: one Dockerfile per component
│   ├── Dockerfile.ort
│   ├── Dockerfile.torch-scatter
│   └── Dockerfile.warpconvnet
├── artifacts.lock.json             # pins + lockhashes + resolved image refs
├── lockhash.sh                     # compute / verify / refresh the lockfile
├── build.sh                        # local & CI driver
└── README.md                       # this file
```

## Architecture

```
                base image (nvcr.io/nvidia/pytorch:26.03-py3)
                              │
                ┌─────────────┼─────────────┬───────────────┐
                ▼             ▼             ▼               ▼
              ort         torch_scatter  warpconvnet     xformers
              (Tier 1 — built from source per component)  (skipped: pip-installed in Tier 2)
                │             │             │               │
                └────┬────────┴─────────────┘               │
                     ▼                                      │
            tao_pytorch_artifacts:<component>-<lockhash>-<arch>
                     │                                      │
                     ▼                                      ▼
                Dockerfile.optimized — COPY --from=$IMAGE /wheels
                     │
                     ▼
                tao_pytorch_base_image:<commit>-<build>-<arch>
```

Tier 1 artifacts are `FROM scratch` images containing only `/wheels/*.whl`.
Tier 2 stages them in via build-arg-resolved `FROM ${ORT_IMAGE}` references and
`pip install --no-deps` of the wheels.

xformers is special: it's a pre-built wheel from NVIDIA's internal GitLab PyPI
registry. No artifact image, no source clone — just `pip install` inside
Tier 2. See the version pin at the top of `Dockerfile.optimized`.

## Lockhash — what makes a cache hit

For each Tier‑1 component:

```
lockhash = sha256( base_digest = live `docker buildx imagetools inspect <base_image>`
                   pin         = artifacts.lock.json .components[$c].pin
                   gpu_archs   = artifacts.lock.json .gpu_archs
                   dockerfile_sha = sha256(component Dockerfile) )[0..16]

image = nvcr.io/nvstaging/tao/tao_pytorch_artifacts:<component>-<lockhash>-<arch>
```

Same inputs → same tag. The build's "cache hit" check is literally
`docker buildx imagetools inspect <tag>`: tag exists in registry → skip the
build, reuse the existing image.

What invalidates the lockhash:

| Change | Invalidates |
|---|---|
| Bump `pin` for one component | that component only |
| Edit `artifacts/Dockerfile.<component>` (any byte) | that component only |
| Bump `PYTORCH_BASE_IMAGE` or NGC re-pushes its tag | all three |
| Change `gpu_archs` | all three |

What does *not* invalidate it:
- `requirements-pip*.txt`, `manifest.json`, `Dockerfile.optimized` (those affect
  Tier 2 only — the artifact builds don't read them)
- xformers version (lives in `Dockerfile.optimized`, not in the lockfile)

## Quick start

### Local single-arch build

```bash
# x86 native, push to registry
./docker/optimized/build.sh --build --x86 --push

# arm64 cross-build, push to registry
./docker/optimized/build.sh --build --arm --push

# x86 native, --load locally instead of push
./docker/optimized/build.sh --build --x86
```

This runs `lockhash --verify`, then for each component: skips if cached,
otherwise builds and pushes the artifact. Then assembles Tier‑2.

### Build only one Tier‑1 artifact (matrix-cell mode)

```bash
./docker/optimized/build.sh --build --x86 --component ort --push
```

Used by the Jenkins matrix; useful locally when iterating on a single
component's Dockerfile.

### Build only Tier‑2 (assume artifacts already pushed)

```bash
./docker/optimized/build.sh --build --x86 --skip-artifacts --push
```

### Refresh the lockfile

```bash
./docker/optimized/lockhash.sh --update
```

Resolves the live `base_digest`, recomputes all lockhashes, writes the lockfile
to disk. Run this after bumping a pin or editing a component Dockerfile.

### Verify lockfile is in sync (CI does this on every run)

```bash
./docker/optimized/lockhash.sh --verify
```

Exit 0 = up to date. Exit 1 = drift; lockfile needs an `--update`.

## Workflow: bumping a component pin

1. Edit `artifacts.lock.json` — bump `.components.<component>.pin`.
2. Run `./docker/optimized/lockhash.sh --update` — recomputes that component's
   `lockhash` and `image` fields.
3. Commit both `artifacts.lock.json` and any code changes.
4. Push. CI's next run sees the new tag is missing in the registry, builds and
   pushes that component, the other two stay cached.

## Workflow: changing a component's build

1. Edit `artifacts/Dockerfile.<component>`.
2. `./docker/optimized/lockhash.sh --update` — `dockerfile_sha` changed →
   new lockhash → new tag.
3. Commit the Dockerfile *and* the lockfile together.
4. Push. CI rebuilds that component on next run.

## Workflow: bumping `PYTORCH_BASE_IMAGE`

1. Edit `artifacts.lock.json` `.base_image` to the new tag.
2. Run `./docker/optimized/lockhash.sh --update`. The live `base_digest`
   resolves to the new content; all three component lockhashes change.
3. Verify xformers compatibility — `Dockerfile.optimized`'s
   `XFORMERS_VERSION` is tied to the base image's torch ABI. The internal
   GitLab PyPI registry publishes wheels tagged `+nv26.3.<pipeline>`,
   `+nv25.10.<pipeline>`, etc. Pick the one matching your new base.
4. Commit lockfile + `Dockerfile.optimized`. CI rebuilds all three artifacts +
   both base images on the next run.

## CI (Jenkins)

The pipeline `jenkinsfile.base-image-optimized` at the repo root drives this.
Stages:

```
validate
└── lockhash_verify                   # drift guard, fails fast on stale lockfile
    └── build artifacts (parallel, 6 cells)
         ├── ort-x86, torch_scatter-x86, warpconvnet-x86
         └── ort-arm64, torch_scatter-arm64, warpconvnet-arm64
              └── build base images (parallel, 2 cells)
                   ├── x86 base image
                   └── arm64 base image
                        └── update_image_tags     # gated by SKIP_MANIFEST_UPDATE param
```

Notes on Jenkins behavior:

- Both base-image cells gate on **all 6** artifact cells. Jenkins declarative
  doesn't allow nested `parallel` blocks, so per-arch failure independence
  isn't possible without restructuring. If any artifact cell fails, neither
  base image is built that run. Successfully-built artifacts still push their
  tags, so the next re-run starts from cache hits for everything that worked.
- `SKIP_MANIFEST_UPDATE` (job parameter, default `false`) gates the
  `update_image_tags` stage. Set `true` for validation runs that shouldn't
  commit digest updates back to the branch.
- The pipeline rejects `BRANCH=main`/`master` because `update_image_tags`
  pushes a commit to `BRANCH`.

## Troubleshooting

**`lockhash.sh: <file> is OUT OF DATE`** — you edited a component Dockerfile
or a pin in `artifacts.lock.json` without running `--update`. Run
`./docker/optimized/lockhash.sh --update` and commit the lockfile.

**`failed to resolve digest for <base_image>`** — `lockhash.sh --update`
needs `docker manifest inspect` access to NGC. Make sure you're logged in
(`docker login nvcr.io`) and that the base image tag actually exists.

**Tier‑1 artifact build OOMs** — buildx's docker-container driver reports
host memory via `/proc/meminfo`, not the build container's cgroup limit. The
parallelism formula in component Dockerfiles can over-provision. Drop
`MAX_JOBS_CAP` (currently `4` for ORT, `8` for the others) on the affected
Dockerfile. Bump it via `--update` after.

**xformers `ImportError: undefined symbol …`** — pinned xformers wheel was
built against a different torch ABI than your base image's torch. Find a
matching `+nv<base_release>.<pipeline>` version on the GitLab index and
update `XFORMERS_VERSION` in `Dockerfile.optimized`. List available versions:

```bash
pip index versions xformers \
    --index-url https://gitlab-master.nvidia.com/api/v4/projects/100660/packages/pypi/simple
```

**Inner `parallel` compile error in Jenkinsfile** — Jenkins declarative
forbids `parallel` inside `parallel`. The pipeline structure is intentionally
flat (6 + 2 top-level cells) to avoid this.

**buildx fails to reach internal hosts** — `--network=host` is set in
`build.sh`'s buildx invocations. If a new Tier‑1 Dockerfile gets added, make
sure `build.sh` continues to pass `--network=host` for that step too.

## Relationship to the legacy build

`docker/Dockerfile` and `docker/build.sh` (one level up) are the legacy
monolithic path. Both still work, both produce the same base image. The
optimized path runs alongside until validated, after which the legacy can be
retired. xformers in *both* paths now uses the GitLab PyPI install (no source
clone in either).
