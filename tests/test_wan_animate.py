"""Tests for the Wan-Animate-2 pass planner and graph builder.

The planner is worth testing hard: the model can only generate a fixed block of
frames per pass, and an off-by-one in how the blocks overlap shows up as a
visible hiccup at every seam rather than as an error, so it will not be caught
by anything crashing.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import wan_animate as wa  # noqa: E402


# ── frame counts the latent packing accepts ──────────────────────────────────


@pytest.mark.parametrize("value", [1, 2, 5, 6, 17, 33, 80, 81, 82, 200])
def test_valid_length_is_one_mod_four(value):
    assert wa.valid_length(value) % 4 == 1


@pytest.mark.parametrize("value", [5, 9, 17, 81])
def test_valid_length_leaves_valid_values_alone(value):
    assert wa.valid_length(value) == value


def test_valid_length_rounds_up_never_down():
    for value in range(1, 200):
        assert wa.valid_length(value) >= min(value, 5)


# ── pass planning ────────────────────────────────────────────────────────────


def test_single_pass_for_a_short_clip():
    segments = wa.plan_segments(81)
    assert len(segments) == 1
    assert segments[0].read_start == 0
    assert segments[0].length == 81
    assert not segments[0].trim_first


def test_short_clip_shrinks_the_only_pass():
    segments = wa.plan_segments(20)
    assert len(segments) == 1
    assert segments[0].length == 21  # rounded up to 1 mod 4
    assert segments[0].length % 4 == 1


def test_second_pass_reads_one_frame_early():
    """The anchor frame is regenerated, so the pass must re-read it."""
    segments = wa.plan_segments(200)
    assert len(segments) >= 2
    second = segments[1]
    assert second.read_start == wa.SEGMENT_FRAMES - 1
    assert second.trim_first


def test_anchor_drift_does_not_accumulate():
    """The failure this guards is silent: no error, just a worse seam each pass.

    Advancing by ``length`` instead of ``length - 1`` looks right for the second
    pass and is wrong by one more frame on every pass after it.
    """
    segments = wa.plan_segments(81 * 12)
    assert len(segments) >= 6
    for index, segment in enumerate(segments[1:], start=1):
        expected = index * (wa.SEGMENT_FRAMES - 1)
        assert segment.read_start == expected, (
            f"pass {index} starts at {segment.read_start}, expected {expected}")


def test_only_the_first_pass_keeps_all_its_frames():
    segments = wa.plan_segments(500)
    assert not segments[0].trim_first
    assert all(segment.trim_first for segment in segments[1:])


def test_continuation_passes_contribute_one_fewer_frame():
    segments = wa.plan_segments(400)
    assert segments[0].new_frames == segments[0].length
    for segment in segments[1:]:
        assert segment.new_frames == segment.length - 1


@pytest.mark.parametrize("total", [1, 17, 80, 81, 82, 161, 162, 300, 1000])
def test_plan_always_covers_the_whole_clip(total):
    segments = wa.plan_segments(total)
    covered = sum(segment.new_frames for segment in segments)
    assert covered >= total, f"{covered} < {total}"


@pytest.mark.parametrize("total", [82, 161, 162, 300, 1000])
def test_plan_does_not_waste_a_whole_extra_pass(total):
    """Coverage must not overshoot by more than the final pass can help."""
    segments = wa.plan_segments(total)
    without_last = sum(s.new_frames for s in segments[:-1])
    assert without_last < total


def test_read_windows_are_contiguous_with_a_one_frame_overlap():
    segments = wa.plan_segments(500)
    for previous, current in zip(segments, segments[1:]):
        previous_end = previous.read_start + previous.length
        assert current.read_start == previous_end - 1


def test_plan_is_bounded_for_absurd_input():
    assert len(wa.plan_segments(10_000_000)) <= 401


# ── output geometry ──────────────────────────────────────────────────────────


def test_fit_dimensions_snaps_to_sixteen():
    width, height = wa.fit_dimensions(1920, 1080)
    assert width % 16 == 0 and height % 16 == 0


def test_fit_dimensions_keeps_aspect_ratio_roughly():
    width, height = wa.fit_dimensions(1080, 1920)
    assert height > width  # portrait in, portrait out
    assert abs((width / height) - (1080 / 1920)) < 0.05


def test_fit_dimensions_respects_the_pixel_budget():
    for source in [(1920, 1080), (1080, 1920), (640, 640), (3840, 2160)]:
        width, height = wa.fit_dimensions(*source, budget=480 * 848)
        assert width * height <= 480 * 848 * 1.15


def test_settings_snap_to_sixteen():
    settings = wa.TransferSettings(width=481, height=854).snapped()
    assert settings.width % 16 == 0
    assert settings.height % 16 == 0


# ── graph construction ───────────────────────────────────────────────────────


def _graph(segment, continue_name=None):
    return wa.build_prompt("ref.png", "pose.mp4", segment,
                           wa.TransferSettings(), 7, continue_name, "out")


def test_first_pass_has_no_anchor():
    graph = _graph(wa.plan_segments(81)[0])
    assert "continue" not in graph
    assert "continue_motion" not in graph["animate"]["inputs"]
    assert graph["animate"]["inputs"]["video_frame_offset"] == 0
    assert "drop_anchor" not in graph
    assert graph["save"]["inputs"]["images"] == ["decode", 0]


def test_continuation_pass_wires_the_anchor_and_drops_it():
    segment = wa.plan_segments(300)[1]
    graph = _graph(segment, continue_name="anchor.png")
    assert graph["continue"]["inputs"]["image"] == "anchor.png"
    assert graph["animate"]["inputs"]["continue_motion"] == ["continue", 0]
    # The node subtracts one internally, so 1 lands it on the clip's first frame.
    assert graph["animate"]["inputs"]["video_frame_offset"] == 1
    assert graph["drop_anchor"]["inputs"]["batch_index"] == ["animate", 4]
    assert graph["save"]["inputs"]["images"] == ["drop_anchor", 0]


def test_sampler_is_wired_to_the_animate_conditioning():
    graph = _graph(wa.plan_segments(81)[0])
    sample = graph["sample"]["inputs"]
    assert sample["positive"] == ["animate", 0]
    assert sample["negative"] == ["animate", 1]
    assert sample["latent_image"] == ["animate", 2]
    assert graph["trim"]["inputs"]["trim_amount"] == ["animate", 3]


def test_distilled_checkpoint_runs_without_guidance():
    """cfg above 1 doubles the cost for a model trained not to need it."""
    graph = _graph(wa.plan_segments(81)[0])
    assert graph["sample"]["inputs"]["cfg"] == 1.0


def test_cache_node_is_wired_between_the_unet_and_sampling():
    graph = _graph(wa.plan_segments(81)[0])
    assert graph["cache"]["inputs"]["model"] == ["unet", 0]
    assert graph["sampling"]["inputs"]["model"] == ["cache", 0]


def test_turning_the_cache_off_removes_the_node():
    """Its "default" dtype is the *largest* cache, so "off" cannot go through it."""
    settings = wa.TransferSettings(cache_dtype=wa.CACHE_OFF)
    graph = wa.build_prompt("r.png", "p.mp4", wa.plan_segments(81)[0],
                            settings, 1, None, "out")
    assert "cache" not in graph
    assert graph["sampling"]["inputs"]["model"] == ["unet", 0]


def test_cache_ram_costs_are_ordered_and_off_is_free():
    assert wa.CACHE_RAM_GB[wa.CACHE_OFF] == 0.0
    assert (wa.CACHE_RAM_GB["int4"] < wa.CACHE_RAM_GB["int8"]
            < wa.CACHE_RAM_GB["default"])


def test_every_link_points_at_a_real_node():
    graph = _graph(wa.plan_segments(300)[1], continue_name="anchor.png")
    for name, node in graph.items():
        for key, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 \
                    and isinstance(value[0], str):
                assert value[0] in graph, f"{name}.{key} -> missing {value[0]}"


def test_length_matches_the_planned_segment():
    for total in (81, 200, 500):
        for segment in wa.plan_segments(total):
            graph = _graph(segment, "a.png" if segment.trim_first else None)
            assert graph["animate"]["inputs"]["length"] == segment.length


def test_seed_reaches_the_sampler():
    graph = wa.build_prompt("r.png", "p.mp4", wa.plan_segments(81)[0],
                            wa.TransferSettings(), 12345, None, "out")
    assert graph["sample"]["inputs"]["noise_seed"] == 12345


def test_negative_prompt_is_not_the_positive_one():
    graph = _graph(wa.plan_segments(81)[0])
    assert graph["negative"]["inputs"]["text"] == wa.NEGATIVE_PROMPT
    assert graph["positive"]["inputs"]["text"] != wa.NEGATIVE_PROMPT


# ── partial downloads ────────────────────────────────────────────────────────


def _write_safetensors(path, payload=b"\x00" * 64):
    """A minimal but structurally real safetensors file."""
    header = json.dumps({
        "weight": {"dtype": "F32", "shape": [4, 4],
                   "data_offsets": [0, len(payload)]}}).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(len(header).to_bytes(8, "little"))
        handle.write(header)
        handle.write(payload)
    return path


def test_complete_safetensors_is_accepted(tmp_path):
    assert wa.safetensors_complete(_write_safetensors(tmp_path / "m.safetensors"))


def test_truncated_safetensors_is_rejected(tmp_path):
    """A download in flight is a real file; it must not count as ready."""
    path = _write_safetensors(tmp_path / "m.safetensors")
    whole = os.path.getsize(path)
    with open(path, "rb") as handle:
        partial = handle.read(whole - 8)
    with open(path, "wb") as handle:
        handle.write(partial)
    assert not wa.safetensors_complete(path)


def test_empty_and_garbage_files_are_rejected(tmp_path):
    empty = tmp_path / "empty.safetensors"
    empty.write_bytes(b"")
    assert not wa.safetensors_complete(str(empty))
    garbage = tmp_path / "garbage.safetensors"
    garbage.write_bytes(b"not a model at all, not even close")
    assert not wa.safetensors_complete(str(garbage))


def test_absent_file_is_rejected(tmp_path):
    assert not wa.safetensors_complete(str(tmp_path / "nope.safetensors"))


def test_missing_models_reports_partial_downloads(tmp_path, monkeypatch):
    base = tmp_path / "models"
    for folder, name in wa.MODEL_FILES.items():
        (base / folder).mkdir(parents=True, exist_ok=True)
        _write_safetensors(base / folder / name)
    monkeypatch.setattr(wa, "models_dir", lambda: str(base))
    assert wa.missing_models() == []

    # Truncate one of them the way an interrupted download would.
    victim = base / "vae" / wa.VAE_NAME
    victim.write_bytes(victim.read_bytes()[:10])
    assert wa.VAE_NAME in wa.missing_models()


# ── reading results back ─────────────────────────────────────────────────────


def test_frames_come_from_the_save_node_not_the_video_preview():
    """LoadVideo reports a one-frame preview under the same "images" key.

    Taking the first node that has any would return that single frame and
    silently truncate a finished render to one frame.
    """
    record = {"outputs": {
        "pose_video": {"images": [{"filename": "preview.png"}],
                       "animated": [False]},
        "save": {"images": [{"filename": f"seg_{i}.png"} for i in range(81)]},
    }}
    assert len(wa._saved_images(record)) == 81


def test_no_frames_when_the_save_node_is_absent():
    assert wa._saved_images({"outputs": {"pose_video": {"images": [{}]}}}) == []
    assert wa._saved_images({}) == []
    assert wa._saved_images({"outputs": {}}) == []


def test_settled_waits_for_outputs_to_appear(monkeypatch):
    """The socket says "done" slightly before the history is written."""
    backend = wa.Backend()
    done = {"images": [{"filename": "a.png"}]}
    calls = {"n": 0}

    def history(_prompt_id):
        calls["n"] += 1
        if calls["n"] < 3:
            return {}  # not written yet
        return {"outputs": {wa.SAVE_NODE: done}}

    monkeypatch.setattr(backend, "history", history)
    monkeypatch.setattr(wa.time, "sleep", lambda _s: None)
    record = backend._settled("p", timeout=10.0)
    assert wa._saved_images(record) == done["images"]
    assert calls["n"] >= 3


def test_settled_gives_up_rather_than_hanging(monkeypatch):
    backend = wa.Backend()
    monkeypatch.setattr(backend, "history", lambda _p: {})
    monkeypatch.setattr(wa.time, "sleep", lambda _s: None)
    assert wa._saved_images(backend._settled("p", timeout=0.01)) == []


def test_settled_returns_early_on_an_error_status(monkeypatch):
    backend = wa.Backend()
    record = {"status": {"status_str": "error"}}
    monkeypatch.setattr(backend, "history", lambda _p: record)
    monkeypatch.setattr(wa.time, "sleep", lambda _s: None)
    assert backend._settled("p", timeout=30.0) == record


# ── generation rate as a speed lever ─────────────────────────────────────────


def test_lower_generation_fps_covers_more_video_for_the_same_work():
    """Cost is per frame, so halving the rate doubles the footage per pass."""
    fast = wa.TransferSettings(fps=16.0, max_seconds=0.0)
    slow = wa.TransferSettings(fps=8.0, max_seconds=0.0)
    # A ten-second clip needs this many driving frames at each rate.
    at_16 = wa.planned_frames(fast, int(10 * 16))
    at_8 = wa.planned_frames(slow, int(10 * 8))
    assert at_8 < at_16
    assert wa.work_units(slow, at_8) < wa.work_units(fast, at_16)


def test_generation_rate_is_a_setting_not_a_constant():
    assert wa.TransferSettings(fps=8.0).fps == 8.0
    assert wa.TransferSettings().fps == wa.DEFAULT_FPS


def test_smoothing_only_applies_when_it_raises_the_rate():
    """Interpolating down would throw frames away for no gain."""
    assert wa.TransferSettings(fps=16.0, output_fps=0.0).output_fps == 0.0
    settings = wa.TransferSettings(fps=8.0, output_fps=24.0)
    assert settings.output_fps > settings.fps


def test_smoothness_choices_are_ordered_and_consistent():
    rates = [gen for _l, gen, _o, _s in wa.SMOOTHNESS_PRESETS]
    assert rates == sorted(rates, reverse=True), "fastest-motion option first"
    for _label, gen, out, saving in wa.SMOOTHNESS_PRESETS:
        assert out == 0.0 or out > gen, "smoothing must raise the rate"
        assert 0 < saving <= 1.0


# ── page wiring ──────────────────────────────────────────────────────────────


def test_nav_indices_match_the_nav_list():
    """Pages are added in NAV_ITEMS order; the constants must agree with it."""
    from modules import ui_nav
    names = [title for _icon, title in ui_nav.NAV_ITEMS]
    assert names[ui_nav.MOTION_PAGE_INDEX] == "Motion"
    assert names[ui_nav.FACES_PAGE_INDEX] == "Faces"
    assert names[ui_nav.ABOUT_PAGE_INDEX] == "About"
    assert len(set(names)) == len(names), "page titles must be unique"


def test_partial_result_reports_itself_as_incomplete():
    partial = wa.TransferResult(path="a.mp4", frames=81, seconds=900.0,
                                segments=3, completed_segments=1,
                                error="Cancelled.")
    assert not partial.complete
    whole = wa.TransferResult(path="a.mp4", frames=243, seconds=2700.0,
                              segments=3, completed_segments=3)
    assert whole.complete


# ── estimation ───────────────────────────────────────────────────────────────


def test_estimate_is_none_before_anything_has_been_measured(monkeypatch):
    monkeypatch.setattr(wa, "_calibration", lambda: {})
    monkeypatch.setattr(wa, "_DEFAULT_SECONDS_PER_STEP", 0.0)
    assert wa.estimate_minutes(wa.TransferSettings(), 81) is None


def _rate(value=1.0):
    return lambda: {"seconds_per_unit": value}


def test_estimate_scales_with_length_once_measured(monkeypatch):
    monkeypatch.setattr(wa, "_calibration", _rate())
    short = wa.estimate_minutes(wa.TransferSettings(), 81)
    long = wa.estimate_minutes(wa.TransferSettings(), 81 * 4)
    assert short is not None and long is not None
    assert long[0] > short[0]


def test_estimate_respects_the_length_cap(monkeypatch):
    monkeypatch.setattr(wa, "_calibration", _rate())
    settings = wa.TransferSettings(max_seconds=5.0)
    capped = wa.estimate_minutes(settings, 10_000)
    uncapped = wa.estimate_minutes(wa.TransferSettings(), 10_000)
    assert capped[1] < uncapped[1]


def test_cost_model_counts_frames_not_just_passes():
    """A short final pass is cheaper than a full one; the model must know.

    Recording a 33-frame render as if it were a whole 81-frame pass made the
    estimate roughly three times too optimistic for every subsequent job.
    """
    settings = wa.TransferSettings(width=480, height=848, steps=10)
    assert wa.work_units(settings, 33) < wa.work_units(settings, 81)
    assert wa.work_units(settings, 81) == pytest.approx(
        wa.work_units(settings, 27) * 3)


def test_cost_model_scales_with_steps_and_pixels():
    small = wa.TransferSettings(width=400, height=704, steps=6)
    big = wa.TransferSettings(width=480, height=848, steps=6)
    assert wa.work_units(big, 81) > wa.work_units(small, 81)
    more_steps = wa.TransferSettings(width=400, height=704, steps=12)
    assert wa.work_units(more_steps, 81) == pytest.approx(
        wa.work_units(small, 81) * 2)


def test_measurement_round_trips_to_a_matching_estimate(tmp_path, monkeypatch):
    """Feeding a measurement back in must predict that same render."""
    monkeypatch.setattr(wa, "_CALIBRATION_FILE", str(tmp_path / "cal.json"))
    settings = wa.TransferSettings(width=480, height=848, steps=6,
                                   max_seconds=5.0)
    frames = wa.planned_frames(settings, 80)
    wa.record_measurement(settings, frames, 925.0)
    low, high = wa.estimate_minutes(settings, 80)
    assert low <= 925.0 / 60.0 <= high


def test_planned_frames_includes_regenerated_anchors():
    settings = wa.TransferSettings(max_seconds=0.0)
    # Two passes: 81 frames plus a second pass that regenerates one of them.
    assert wa.planned_frames(settings, 161) == sum(
        s.length for s in wa.plan_segments(161))
