# Morphify

Real-time face swap with **virtual camera output**: the processed webcam feed
is published back to the system as a camera device, so Discord, Zoom, Teams,
OBS and browsers can use it like any other webcam.

Morphify is built on the open-source
[Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) project and is
released under the same licence, the **AGPL-3.0**. Credit for the underlying
face-swap engine belongs to that project and its contributors; what is added
here is the virtual camera, the interface, the packaging and the fixes listed
at the end.

---

## Running from source

```bash
venv\Scripts\python.exe run.py --execution-provider cuda
```

First run downloads ~1.5 GB of models into `models/`, or run it up front:

```bash
venv\Scripts\python.exe setup\download_models.py
```

Check the install at any time — this is the fastest way to see what is wrong:

```bash
venv\Scripts\python.exe run.py --self-test
```

## Building the app

```bash
venv\Scripts\python.exe packaging\build.py --clean --installer
```

Produces `dist/Morphify/Morphify.exe` (~7 GB one-folder bundle) and
`dist/Morphify-1.0.0-Setup.exe`. Requires Inno Setup 6 in
`tools/innosetup/`.

The bundle doubled when portrait animation was added: it is the only feature
that needs PyTorch-CUDA, and torch plus torchvision plus their CUDA libraries
account for the difference.

---

## What it does

**Live** — two modes, plus the transport controls.

* **Face swap** — replaces the face only. Fastest (~31 fps at 960x540 on an
  RTX 3060) and the most robust to head movement.
* **Portrait animation** — drives a still photo with your movement. The head,
  hair and expression are *generated* rather than pasted, so it is photoreal
  and seamless. Heavier: about 10 fps, and only the head moves.

Also: virtual camera, recording, snapshots, a bypass ("panic") switch that
drops the swap instantly without dropping the stream, a split before/after
view, and a Performance/Balanced/Quality preset that sets capture size,
detection rate and enhancer in one go.

### Portrait animation

Press **Set neutral pose** facing the camera straight on: every later frame
is measured as a *change* from that reference, which is what lets one
person's movement drive a different person's face without forcing your head
shape onto theirs.

Only the head is generated — the model has no notion of a body. There is no
photoreal full-body equivalent and there cannot be on this hardware:
re-posing a whole body convincingly needs a diffusion model, which is 20–50
network passes per frame rather than one.

A **Full Takeover** mode (transplanting hair, skin tone and background onto
the plain swap) was built and withdrawn. Hair transplanted as a warped 2D
cutout read as a sticker with visible seams — the wrong technique rather
than a tuning problem. `modules/takeover.py` remains in the tree, unused.

**Faces** — your source-face library. Star favourites (they sort first),
search, rename, delete, and step through faces live with `[` and `]`.

*Find faces* searches Wikimedia Commons and Openverse, runs the face detector
over every result, and **Select faces** picks the ones that actually contain a
face in one click. Results are saved under the search term
(`kai-cenat-01.jpg`), which is what makes the library searchable later. You
can also generate synthetic portraits of people who do not exist, or paste a
direct image link.

**Identity blending** — photos sharing a name group are averaged into one
identity, so the swap follows the *person* rather than one photo's lighting
and angle. Web results are not reliably the same person, so embeddings that
disagree with the group are rejected rather than blended in: on a real
"kai cenat" search, genuine photos scored 0.74–0.82 against the group centre
and a stray LeBron photo scored 0.30, which is well clear of the 0.35 cut.

**Studio** — swap a face onto an image or video file, and **Motion
transfer**.

### Motion transfer (offline)

Give it a photo of someone and a video of you moving; it renders that person
performing your motion — whole body, photoreal, generated rather than warped.

    your video ------> raw driving frames -> motion, expression, hands
    reference photo -> identity + CLIP vision
    prompt ---------> appearance and background
                       Wan-Animate-2 14B (distilled, int8) -> mp4

**It is not live and cannot be.** The model generates 81 frames as one block
and cannot emit the first until it has finished all of them, so there is no
streaming form of it at any GPU budget. This is a property of the model, not
of the hardware.

#### Why this is the second attempt

The first version used **Wan 2.1 VACE 1.3B** driven by extracted OpenPose
skeletons. It was rejected in testing, correctly. Two things were wrong with
it and only one was a bug:

1. **A skeleton is a lossy control signal.** It keeps joint positions and
   throws away face, hands and everything about how a body actually moves.
   Animate-2 conditions on the *raw driving frames* instead, which is why it
   holds expression and hand detail that the skeleton path could not.
2. **1.3B against 14B.** Roughly eleven times the parameters. Draft-grade
   identity at 1.3B was not a tuning problem.

`modules/motion_transfer.py` (the VACE path) is still in the tree but is no
longer reachable from the UI.

#### How arbitrary length works

This is the part the reference ComfyUI workflow makes you do by hand — its own
note tells you to copy-paste a subgraph per 81 frames and rewire
`continue_motion` and `video_frame_offset` yourself. `plan_segments` computes
it instead.

The subtlety is that a continuation pass takes the previous pass's last frame
as a temporal anchor, **regenerates it**, and discards the copy. So it advances
by `length - 1`, not `length`:

    pass 0:  reads driving[0   .. 81 )   keeps all 81
    pass 1:  reads driving[80  .. 161)   keeps 80, drops the anchor
    pass 2:  reads driving[160 .. 241)   keeps 80
             ^ +80 per pass, not +81

Getting this wrong raises nothing. It leaves the driving video one frame
further ahead of the anchor on every pass, so the seams stutter progressively
harder the longer the clip. It was wrong in the first draft here and is now
covered by `test_anchor_drift_does_not_accumulate`.

#### Settings that matter

- **cfg is 1.0.** The distilled checkpoint is trained not to need classifier-
  free guidance, so each step is one model pass rather than two. Raising it
  doubles the cost for nothing.
- **16 fps.** Wan was trained around it. Driving a 30 fps clip costs nearly
  twice as much for motion the model cannot resolve anyway, so the clip is
  resampled to 16 and played back at 16 — same duration, close to half the
  work.
- **Pose-branch cache.** Runs the pose branch once per pass instead of once
  per step, roughly a 2x speedup, paid for in system RAM (~12.5 GB at
  480x832/81 frames in bf16, halved by int8, quartered by int4).
- **Prompt describes looks, never motion.** The background is generated from
  the text and is decoupled from both inputs. Motion goes in the second box.

#### Measured on the RTX 3060 12 GB

Real numbers from a full 81-frame render at 480x848, not extrapolations:

| Setting | Per sampler step | 5 s clip (81 frames) |
|---|---|---|
| Draft, 6 steps | **140 s** | **15.4 min** (measured) |
| Normal, 10 steps | 140 s | ~21-35 min |
| 15 s at 10 steps | 140 s | ~62-104 min (3 passes) |

Two configuration faults cost roughly a **6x** slowdown between them, and both
looked like "the model is just slow" until measured:

1. **Pinned memory.** The backend pins staged weights by default. Pinned pages
   cannot be swapped, so with a 15.5 GB checkpoint on a 24 GB machine Windows
   paged out everything *else*; free RAM fell to 1.8 GB and a single step did
   not finish in twenty minutes. `--disable-pinned-memory --cache-none
   --fast-disk` streams from the NVMe instead, which reads at GB/s.
2. **`torch` built against cu129.** This checkpoint is int8/convrot quantized
   and its fused kernels need cu130 — the backend logs `You need pytorch with
   cu130 or higher` and reports `comfy_kitchen backend cuda: disabled: True`,
   falling back to an unfused eager path. Upgrading the backend venv to
   `torch 2.14.0+cu130` took a step from **533 s to 140 s, a 3.8x speedup**.

Keep the backend venv on a cu130-or-newer build. There is no smaller checkpoint
to fall back on: the repo's only other diffusion weights are the 30.5 GB bf16
ones, so 15.5 GB is the floor.

#### Requirements

The checkpoint is **15.5 GB and this machine has 12 GB of VRAM**, so weights
stream from system RAM every step. That makes free RAM the binding constraint,
not VRAM, and it is why the panel refuses to start when a game is holding
either pool — it would thrash rather than merely run slower.

Models total **23 GB** and are **not** bundled in the installer. They are
validated with `safetensors_complete`, which derives the expected size from the
file's own header, because `os.path.isfile` is true for a download still in
flight and would otherwise let a render start and fail deep into it.

#### Backend

The 14B model runs in a **private ComfyUI checkout** at `E:\morphify-backend`,
started on demand and reused between renders (loading 15.5 GB is too expensive
to repeat per clip). It is deliberately separate from any ComfyUI already
installed on the machine; the two share only a models folder through
`extra_model_paths.yaml`, so an existing install cannot be disturbed.
Morphify generates the API graph itself rather than shipping a workflow file —
see `build_prompt` — and tracks sampler steps over the websocket for progress.

**Setup** — virtual camera, capture size and rate, execution provider, the
NSFW filter, and where the models live.

Recordings and snapshots land in `captures/`. Closing the window while the
feed is live hides to the tray instead of quitting, since the feed is usually
being used by a call.

### Keyboard shortcuts

| | | | |
|---|---|---|---|
| `Ctrl+L` | Go live / stop | `]` | Next face |
| `Ctrl+K` | Virtual camera | `[` | Previous face |
| `Ctrl+R` | Record | `Ctrl+1..5` | Switch page |
| `Ctrl+S` | Snapshot | `Ctrl+D` | Split view |
| `Ctrl+B` | Bypass (panic) | `Ctrl+M` | Mirror preview |

The About page renders this list from the same table that binds the keys, so
it cannot drift.

---

## Virtual camera

Windows needs a registered DirectShow filter to expose a virtual camera. The
app drives the one that ships with **OBS Studio**, which has to be installed
once — OBS itself never has to run. Other applications then see a camera
called *OBS Virtual Camera*.

Setup → Virtual camera reports whether the filter is present, and the
installer offers the download link if it is missing.

Verify the whole path end to end:

```bash
venv\Scripts\python.exe setup\test_virtualcam.py
```

That publishes a test pattern, opens the virtual camera as a capture device,
and checks the frames coming back match.

---

## Where things are stored

Running from a source checkout everything stays in the project folder.
An installed build writes to `%LOCALAPPDATA%\Morphify\`:

| What | Path |
|---|---|
| Models | `models/` — relocatable via Setup → Models → Change folder |
| Face library | `faces/` |
| Settings | `switch_states.json` |
| Model location override | `config.json` |
| Self-test report | `self-test.log` |

The models are ~1.5 GB and need ~2.4 GB free during download. If the system
drive is short on space, point them at another drive from Setup — the app
checks before downloading rather than filling the volume and failing part
way.

---

## Architecture notes

```
camera ─▶ [capture thread] ─▶ queue ─▶ [processing thread] ─┬─▶ preview sink
                                                            └─▶ virtual camera sink
```

* **`modules/live_engine.py`** — the threaded pipeline, with no Qt
  dependency. Processed frames fan out to any number of registered sinks;
  neither the preview nor the virtual camera can back-pressure the swap loop.
  Adding a new consumer (a recorder, a second preview) means implementing
  `send(frame)` and calling `add_sink`.
* **`modules/virtual_camera.py`** — the DirectShow sink. Resolution is locked
  when the device opens, so frames are letterboxed rather than forcing a
  reopen, which consumers would see as the device dropping.
* **`modules/ui.py`** — the window. `modules/ui_theme.py` holds the palette
  and stylesheet, `modules/ui_common.py` the shared helpers,
  `modules/ui_dialogs.py` the mapper and preview windows.
* **`modules/gpu_paths.py`** — registers the CUDA/cuDNN DLL directories from
  the pip wheels. Imported by `modules/__init__.py` so it runs no matter
  which entry point starts the app.
* `modules/recorder.py` — the recording sink, and the clearest example
  of how to add a new consumer of the live feed.
* `modules/motion_transfer.py` — offline Wan VACE rendering;
  `modules/body_pose.py` supplies the skeletons, `modules/ui_motion.py` the
  panel.
* `modules/live_portrait.py` — withdrawn portrait animation. The only part
  of the app that uses PyTorch, and only because onnxruntime has no CUDA
  kernel for 5D `grid_sample` or 5D `Resize`; see the module docstring.
* `modules/takeover.py` — withdrawn hair/skin/background transplant, kept
  for reference.
* `modules/face_identity.py` — multi-photo identity blending with outlier
  rejection.
* `modules/image_search.py` — key-free image search providers.
* `modules/ui_legacy.py` is the original single-column UI, kept for
  reference only. It is excluded from the build.

---

## Changes to the upstream code

Fixes made while getting this working, in rough order of impact:

1. **CUDA only worked via `run.py`.** The DLL-directory registration lived in
   that script, so tests, tooling and a frozen build silently fell back to
   CPU. Moved to `modules/gpu_paths.py`, imported from `modules/__init__.py`.
2. **`requirements.txt` pinned `onnxruntime-gpu==1.26.0`, which does not
   exist on PyPI** (latest is 1.23.2) — the install could not succeed as
   written.
3. **The FP16 swap model was gated behind a torch-CUDA probe.** Model
   selection and the CUDA-graph session are pure onnxruntime concerns, so
   requiring a ~2.5 GB CUDA torch build to unlock them was wasteful. Added
   `_has_ort_cuda()`; `_HAS_TORCH_CUDA` still guards the torch blend paths
   that genuinely need it.
4. **The enhancer model URLs 404'd.** They pointed at a `GPEN-BFR` release
   tag; the assets live under `Models`.
5. **The GFPGAN enhancer required a file named `gfpgan-1024.onnx` that has no
   published source.** The alignment size is read from the model's own input
   shape, so any square GFPGAN export works — it now accepts several names.
6. **A tightly cropped source portrait silently produced no swap.**
   RetinaFace cannot detect a face that fills the whole frame, which is
   exactly what an avatar or the Random Face button gives you. Added
   `get_source_face()`, which retries with margin.
7. **The Random Face button saved an HTML page as a `.jpg`** —
   thispersondoesnotexist.com moved the image off its root URL.
8. **The app refused to launch without ffmpeg**, even though live webcam mode
   never touches it. Now a warning; the video paths check where it matters.
9. **Installed builds would have written models and settings into Program
   Files.** Paths are now install-aware.
10. **The Random Face button only worked once.** It rewrote a single temp
    file, and the live loop cached the source face keyed on the *path*, so
    every face after the first was ignored while the thumbnail updated. Faces
    now get unique names, and the cache is keyed on the file's size and
    modification time so replacing an image in place is noticed too.
    Covered by `tests/test_source_token.py`.
11. **`cv2.VideoWriter.isOpened()` lies.** It reports success for paths it
    cannot write, which would have made a failed recording look like a
    working one. The recorder verifies the file actually appeared.
12. **`get_source_face()` returned coordinates in the wrong space.** When it
    fell back to detecting on a padded copy, the bbox and keypoints stayed in
    padded coordinates — a bbox running to y=1118 in a 1024px image. Harmless
    while only the embedding was used; it put the transplanted hair in orbit
    around the head the moment geometry mattered. Coordinates are now
    translated back.
13. **The published LivePortrait model will not load in stock onnxruntime.**
    It uses a non-standard `GridSample3D` op, which upstream works around
    with a custom onnxruntime build and a TensorRT plugin. Two rewrites were
    needed and neither was enough on its own:
    * `GridSample3D` -> standard `GridSample` (opset 20 added 5D support),
    * 5D `Resize` -> 4D, since only H and W are scaled.

    Even then onnxruntime has no CUDA kernel for 5D `GridSample`, leaving it
    at 637 ms a frame. The shipping path converts the graph to PyTorch with
    `onnx2torch` instead, registering a `GridSample` converter that library
    does not provide, then runs it fp16 + TorchScript: **189 ms -> 83 ms**.
14. **Startup crashed in the installed build after a while.**
    `face_swapper.pre_check()` downloaded the 554 MB FP32 model
    *synchronously at startup*, before any window existed, and a windowed
    PyInstaller build has `sys.stdout = None`, so tqdm's progress bar raised
    `'NoneType' object has no attribute 'write'`. Two faults in one line:
    pre_check now only verifies, and `modules/__init__.py` guarantees both
    streams are writable for every entry point.
15. **The first working render took 18.7 minutes, of which 17 were the VAE
    decode.** Generation itself was 55 seconds. The VAE had been pinned to
    fp32 with tiling on the theory that Wan's VAE shows artefacts in half
    precision; at 480p on this decoder that is not visible, and the cost was
    20x the actual generation. bf16 without tiling (tiling kept only as the
    low-memory fallback) brought the same render to 7.5 minutes.
16. **The first render also ignored the reference photo's setting entirely.**
    The default prompt asked for "cinematic lighting, high quality video" and
    the model obliged, inventing a modern scene. The text prompt competes
    with the reference image, so the default now describes preservation and
    nothing else.
17. **torchvision's compiled extension was silently missing from the
    build.** Recent releases renamed it `_C_stable.pyd`, which PyInstaller's
    hook does not know about and `collect_dynamic_libs` skips because it only
    matches `.dll`. Portrait animation failed with `operator torchvision::nms
    does not exist` — but only in the frozen build, which is why the
    self-test now runs the portrait graph rather than merely importing it.

---

## Responsible use

This makes deepfakes. If you use a real person's face, get their consent and
label the output as synthetic when you share it. Impersonation, harassment
and non-consensual imagery are illegal in many jurisdictions. The NSFW filter
is still wired in and can be enabled in Setup.

### Identity and source-image resolution (measured)

Numbers from `w600k_r50` (ArcFace) on this machine, not estimates.

**The swapper is already "them" by the numbers.** A plain swap scores **0.884**
cosine to the source identity with the Likeness slider at 0. It still does not
*look* like them, because inswapper replaces only the inner face and warps it
onto the target's keypoints — jaw, head shape, hairline and hair stay the
target's at every setting. Perceived likeness is dominated by geometry, which
no identity control reaches.

**Likeness slider** (`modules.globals.identity_strength`,
`face_swapper.strengthened_source`):

| Likeness | to source | to target |
|---|---|---|
| 0% | 0.884 | +0.034 |
| 40% | 0.866 | -0.129 |
| 100% | 0.829 | -0.236 |

It strips the target's identity steadily, but similarity to the *source* falls
too — so past ~40% it is distortion, not likeness. 20-40% is the useful range.

**The trap it avoids:** inswapper computes `latent = normed_embedding . emap`
then divides by that vector's norm, so *scaling* the embedding is a complete
no-op. Strength has to change the embedding's direction (extrapolate along
`source - target`), never its magnitude. Pinned by
`tests/test_identity_strength.py::test_scaling_the_embedding_would_have_been_a_no_op`.

**Source resolution.** ArcFace runs at 112x112 and inswapper at 128x128, so
detail above ~112 px of face width is discarded. Identity retained against a
512 px reference:

| face width | 512 | 160 | 112 | 96 | 64 | 48 | 32 |
|---|---|---|---|---|---|---|---|
| cosine | 0.998 | 0.984 | 0.979 | 0.970 | 0.911 | 0.868 | 0.662 |

Upscaling a source face that is already over ~112 px cannot help. Below ~64 px
it degrades fast and is worth rescuing.

#### Do not "enhance" a small source photo

The obvious repair for a low-resolution source face is a restoration model.
Measured across four identities, every one of them is **worse than plain
bicubic at every size**:

| face width | bicubic | GFPGAN | GPEN-512 |
|---|---|---|---|
| 96 px | **0.966** | 0.729 | 0.938 |
| 64 px | **0.924** | 0.457 | 0.865 |
| 48 px | **0.821** | 0.392 | 0.707 |
| 32 px | **0.676** | 0.305 | 0.439 |

Restoration models invent a plausible, *generic* face rather than recovering
the real one — sharper to the eye, further from the person, which is exactly
the wrong trade when the point is preserving an identity.

A single-face pilot appeared to show GFPGAN winning at 64 px and 32 px. That
was noise; averaging four identities reversed it. One sample is not a result.

So there is nothing to build here beyond warning early, which
`modules/face_quality.py` does when a source face is picked — the detection has
already run at that point, so the check is free.
