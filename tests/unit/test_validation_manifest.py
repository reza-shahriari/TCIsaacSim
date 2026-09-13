"""The validation-data index (ME.1): the fields that gate a measurement, and the refusals.

This index is the reason a statistic from public video can be trusted at all, so the tests are
about the things that would let an untrustworthy one through: a set with no recorded signal path,
an analyser run on data that cannot support it, a licence silently assumed to be permissive, and a
README that has drifted from the fields the code actually reads.

docs/physics-model.md §15 T4; ADR 0003.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from irsim_eval.fetch import Action, download, plan, sha256_of, target_dir, verify
from irsim_eval.manifest import (
    UNSTATED,
    Dataset,
    Manifest,
    load_manifest,
    manifest_path,
    readme_path,
    render_readme,
)


@pytest.fixture(scope="module")
def manifest() -> Manifest:
    return load_manifest()


def decisions(actions: list[Action]) -> dict[str, str]:
    return {a.name: a.decision for a in actions}


# --- the fields that gate a measurement ---------------------------------------------------------


def test_every_set_records_a_licence_and_a_signal_path(manifest: Manifest) -> None:
    """The ME.1 criterion. A set that cannot say what its frames are cannot be measured."""
    for name, dataset in manifest.datasets.items():
        assert dataset.licence.strip(), name
        assert dataset.signal_path.strip(), name
        assert dataset.analysers, name
        assert dataset.why.strip(), name


def test_the_index_names_exactly_one_primary_set(manifest: Manifest) -> None:
    """Two primaries would invite a comparison that averages sets with different signal paths."""
    name, primary = manifest.primary
    assert name == "halmstad_drone_detection"
    assert primary.licence_known
    with pytest.raises(ValueError, match="exactly one primary"):
        Manifest.model_validate(
            {
                "schema_version": 1,
                "checked_utc": "2026-09-13",
                "datasets": {
                    k: {**v.model_dump(exclude_none=True), "role": "primary"}
                    for k, v in list(manifest.datasets.items())[:2]
                },
            }
        )


def test_only_the_y16_set_may_carry_a_noise_decomposition(manifest: Manifest) -> None:
    """A 3-D noise decomposition needs the sensor's own stream, not an ISP's output.

    This is the index earning its place: the same flat-sky patch is a laboratory measurement on a
    Y16-derived clip and a measurement of somebody's unknown AGC on a display-output one. Only the
    Halmstad set states a signal path that supports it.
    """
    assert manifest.usable_for("noise_3d") == ["halmstad_drone_detection"]
    assert manifest.usable_for("temporal_psd") == ["halmstad_drone_detection"]


def test_an_agc_signature_may_not_be_measured_on_the_set_that_has_no_agc(
    manifest: Manifest,
) -> None:
    """The Halmstad clips never went through an AGC, so its histogram shape is the recorder's."""
    halmstad = manifest.datasets["halmstad_drone_detection"]
    assert not halmstad.may_run("agc_signature")
    assert "agc_signature" in halmstad.excluded_analysers
    assert manifest.usable_for("agc_signature") == ["anti_uav_410"]


def test_an_exclusion_beats_an_inclusion(manifest: Manifest) -> None:
    """Belt and braces: a name in both lists must be refused, not allowed."""
    both = manifest.datasets["halmstad_drone_detection"].model_copy(
        update={"analysers": ["noise_3d", "agc_signature"], "excluded_analysers": ["agc_signature"]}
    )
    assert both.may_run("noise_3d")
    assert not both.may_run("agc_signature")


def test_the_range_labelled_set_is_the_only_one_that_can_do_size_versus_range(
    manifest: Manifest,
) -> None:
    """Without a range label, size-versus-range is unmeasurable however many frames there are."""
    assert manifest.usable_for("size_vs_range") == ["lrddv3"]


def test_single_frame_sets_are_priors_and_nothing_temporal(manifest: Manifest) -> None:
    """One frame cannot show an FFC freeze, a temporal PSD or a smear, whatever else it shows."""
    for name in ("irstd_1k", "nuaa_sirst"):
        dataset = manifest.datasets[name]
        assert dataset.role == "prior"
        for analyser in ("ffc_freeze", "temporal_psd", "noise_3d", "size_vs_range"):
            assert not dataset.may_run(analyser), f"{name} must not claim {analyser}"


# --- licences -----------------------------------------------------------------------------------


def test_unstated_is_recorded_rather_than_assumed(manifest: Manifest) -> None:
    """Five of the six publishers stated no terms, and the index says so in those words."""
    unstated = {k for k, v in manifest.datasets.items() if not v.licence_known}
    assert unstated == {"anti_uav_410", "cst_anti_uav", "lrddv3", "irstd_1k", "nuaa_sirst"}
    assert all(manifest.datasets[k].licence == UNSTATED for k in unstated)


def test_a_set_with_no_stated_licence_is_refused_by_default(manifest: Manifest) -> None:
    """Not an error -- a decision handed back to a person, with the reason attached."""
    actions = plan(manifest, ["anti_uav_410"])
    assert decisions(actions) == {"anti_uav_410": "refused"}
    assert "no licence" in actions[0].reason
    assert actions[0].needs_a_person


def test_the_refusal_can_be_overridden_deliberately(manifest: Manifest) -> None:
    """Once a person has read the terms, the flag says so and the set proceeds as manual."""
    actions = plan(manifest, ["anti_uav_410"], accept_unstated_licence=True)
    assert decisions(actions) == {"anti_uav_410": "manual"}


def test_the_licence_gate_is_checked_before_the_access_mode(manifest: Manifest) -> None:
    """An unreleased, unlicensed set reports the licence: that decides whether to want it."""
    assert decisions(plan(manifest, ["cst_anti_uav"])) == {"cst_anti_uav": "refused"}
    assert decisions(plan(manifest, ["cst_anti_uav"], accept_unstated_licence=True)) == {
        "cst_anti_uav": "unreleased"
    }


def test_the_licensed_set_is_still_not_downloaded_automatically(manifest: Manifest) -> None:
    """CC0 does not make a Zenodo record a direct URL; a human still fetches it."""
    actions = plan(manifest, ["halmstad_drone_detection"])
    assert decisions(actions) == {"halmstad_drone_detection": "manual"}
    assert "by hand" in actions[0].reason


def test_an_unknown_set_name_is_an_error_not_a_silent_skip(manifest: Manifest) -> None:
    with pytest.raises(KeyError, match="not in the index"):
        plan(manifest, ["there_is_no_such_dataset"])


# --- fetching and hashing -----------------------------------------------------------------------


def test_a_direct_url_is_planned_as_a_download(manifest: Manifest) -> None:
    """No indexed set has one today, so the path is exercised on a synthetic entry."""
    direct = manifest.datasets["halmstad_drone_detection"].model_copy(
        update={"access": "direct", "download_url": "https://example.invalid/x.zip"}
    )
    synthetic = Manifest(schema_version=1, checked_utc="2026-09-13", datasets={"synthetic": direct})
    actions = plan(synthetic)
    assert decisions(actions) == {"synthetic": "download"}
    assert actions[0].url == "https://example.invalid/x.zip"


def test_download_refuses_an_unencrypted_scheme(tmp_path: pathlib.Path) -> None:
    """These archives are huge; silently downgrading the transport is not ours to decide."""
    with pytest.raises(ValueError, match="non-https"):
        download("http://example.invalid/x.zip", tmp_path / "x.zip")
    with pytest.raises(ValueError, match="non-https"):
        download("ftp://example.invalid/x.zip", tmp_path / "x.zip")


def test_download_streams_a_file_url_byte_for_byte(tmp_path: pathlib.Path) -> None:
    """The transfer path itself, checked without a network: file:// is a real urllib scheme."""
    source = tmp_path / "source.bin"
    payload = bytes(range(256)) * 5000  # larger than one chunk
    source.write_bytes(payload)
    out = download(source.as_uri(), tmp_path / "out" / "copy.bin")
    assert out.read_bytes() == payload
    assert sha256_of(out) == sha256_of(source)


def test_verify_matches_a_recorded_hash_and_reports_a_mismatch(
    manifest: Manifest, tmp_path: pathlib.Path
) -> None:
    """The point of recording a hash: 'measured on these exact bytes' stays checkable."""
    path = tmp_path / "clip.bin"
    path.write_bytes(b"thermal video, honestly")
    digest = sha256_of(path)

    unrecorded = manifest.datasets["halmstad_drone_detection"]
    assert unrecorded.sha256 is None
    ok, seen = verify(unrecorded, path)
    assert ok and seen == digest, "nothing recorded yet, so nothing to contradict"

    recorded = unrecorded.model_copy(update={"sha256": digest})
    assert verify(recorded, path) == (True, digest)

    path.write_bytes(b"thermal video, edited")
    matched, changed = verify(recorded, path)
    assert not matched and changed != digest


def test_a_malformed_hash_is_rejected_by_the_schema() -> None:
    """A truncated or non-hex digest in the index is worse than none: it looks authoritative."""
    base = load_manifest().datasets["halmstad_drone_detection"].model_dump(exclude_none=True)
    for bad in ("abc", "z" * 64, "AB" * 31):
        with pytest.raises(ValueError, match="64 hex"):
            Dataset.model_validate({**base, "sha256": bad})
    good = "a" * 64
    assert Dataset.model_validate({**base, "sha256": good.upper()}).sha256 == good


def test_datasets_land_outside_the_repository_tree_but_under_data(manifest: Manifest) -> None:
    """`data/validation/<set>/`, which .gitignore excludes -- the frames never enter git."""
    destination = target_dir("halmstad_drone_detection")
    assert destination.parent == manifest_path().parent
    assert destination.name == "halmstad_drone_detection"
    ignore = (manifest_path().parents[2] / ".gitignore").read_text()
    assert "data/validation/*/" in ignore


# --- the generated index ------------------------------------------------------------------------


def test_the_readme_is_generated_and_current(manifest: Manifest) -> None:
    """A prose index that disagrees with the machine-readable one is worse than having neither.

    Regenerate with `python scripts/fetch_validation_data.py --render-readme`.
    """
    assert readme_path().read_text(encoding="utf-8") == render_readme(manifest)


def test_the_readme_states_every_licence_and_signal_path(manifest: Manifest) -> None:
    """What a reader must not have to dig for: the terms, and what the frames actually are."""
    text = readme_path().read_text(encoding="utf-8")
    assert "GENERATED FILE" in text
    for name, dataset in manifest.datasets.items():
        assert f"`{name}`" in text
        assert dataset.licence in text
        assert dataset.signal_path.strip().split("\n")[0][:40] in text.replace("\n", " ")


def test_the_manifest_file_is_the_one_the_package_ships(manifest: Manifest) -> None:
    """The loader must find the committed index, not a copy that drifted."""
    raw = yaml.safe_load(manifest_path().read_text(encoding="utf-8"))
    assert set(raw["datasets"]) == set(manifest.datasets)
    assert raw["schema_version"] == manifest.schema_version
