"""Unit tests for ``training.artifacts.ArtifactStore``.

Uses a small in-memory fake S3 client (dependency-injected via
``ArtifactStore(client=...)``) instead of monkeypatching ``boto3`` internals —
see module docstring of ``training/artifacts.py``. The one test that talks to
real Cloudflare R2 lives in ``test_artifacts_r2_integration.py``.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from training.artifacts import (
    R2_ENV_VARS,
    ArtifactStore,
    ValSample,
    _load_r2_env,
    publish_results,
)


class FakeS3Client:
    """Minimal in-memory stand-in for ``boto3.client("s3")``.

    Stores uploaded objects in ``objects`` keyed by ``"<bucket>/<key>"``.
    """

    def __init__(self, *, fail_keys: set[str] | None = None) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_keys = fail_keys or set()

    def _check(self, bucket: str, key: str) -> str:
        path = f"{bucket}/{key}"
        if path in self.fail_keys:
            raise RuntimeError(f"synthetic failure uploading {path}")
        return path

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict:
        path = self._check(Bucket, Key)
        self.objects[path] = bytes(Body)
        return {}

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        path = self._check(Bucket, Key)
        self.objects[path] = Path(Filename).read_bytes()

    def copy_object(self, *, Bucket: str, Key: str, CopySource: dict[str, str]) -> dict:
        destination = self._check(Bucket, Key)
        source = f"{CopySource['Bucket']}/{CopySource['Key']}"
        self.objects[destination] = self.objects[source]
        return {}

    def list_objects_v2(self, *, Bucket: str, Prefix: str, ContinuationToken: str = "0") -> dict:
        """Pages of two keys, so callers must follow ``NextContinuationToken``."""
        keys = sorted(k for k in self.objects if k.startswith(f"{Bucket}/{Prefix}"))
        start = int(ContinuationToken)
        page = keys[start : start + 2]
        contents = [
            {
                "Key": k[len(Bucket) + 1 :],
                "Size": len(self.objects[k]),
                "LastModified": datetime(2026, 11, 26, tzinfo=UTC),
            }
            for k in page
        ]
        more = start + 2 < len(keys)
        return {"Contents": contents, "IsTruncated": more, "NextContinuationToken": str(start + 2)}

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        Path(Filename).write_bytes(self.objects[f"{Bucket}/{Key}"])


# ─── upload_checkpoint ──────────────────────────────────────────────────────


def test_upload_checkpoint_writes_expected_key(tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"fake-checkpoint-bytes")
    client = FakeS3Client()
    store = ArtifactStore(experiment_name="exp1", client=client, enabled=True)

    uri = store.upload_checkpoint(ckpt)

    expected_key = "ml-data/artifacts/exp1/checkpoints/best.ckpt"
    assert uri == f"r2://{expected_key}"
    assert client.objects[expected_key] == b"fake-checkpoint-bytes"


def test_upload_checkpoint_custom_bucket_and_prefix(tmp_path):
    ckpt = tmp_path / "ep3_mse_0.1234.ckpt"
    ckpt.write_bytes(b"x")
    client = FakeS3Client()
    store = ArtifactStore(
        experiment_name="exp2", bucket="other-bucket", prefix="/runs/", client=client, enabled=True
    )

    uri = store.upload_checkpoint(ckpt)

    assert uri == "r2://other-bucket/runs/exp2/checkpoints/ep3_mse_0.1234.ckpt"


def test_alias_checkpoint_copies_inside_r2_without_reupload(tmp_path):
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"selected-state")
    client = FakeS3Client()
    store = ArtifactStore(experiment_name="exp", client=client, enabled=True)
    store.upload_checkpoint(ckpt)

    uri = store.alias_checkpoint("last.ckpt", "best_real_r3.ckpt")

    key = "ml-data/artifacts/exp/checkpoints/best_real_r3.ckpt"
    assert uri == f"r2://{key}"
    assert client.objects[key] == b"selected-state"


def test_upload_checkpoint_disabled_is_noop(tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"x")
    client = FakeS3Client()
    store = ArtifactStore(experiment_name="exp1", client=client, enabled=False)

    assert store.upload_checkpoint(ckpt) is None
    assert client.objects == {}


def test_upload_checkpoint_exception_does_not_propagate(tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"x")
    client = FakeS3Client(fail_keys={"ml-data/artifacts/exp1/checkpoints/best.ckpt"})
    store = ArtifactStore(experiment_name="exp1", client=client, enabled=True)

    # Must not raise.
    result = store.upload_checkpoint(ckpt)
    assert result is None


# ─── upload_file ────────────────────────────────────────────────────────────


def test_upload_file_writes_expected_key(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_bytes(b'{"si_sdr": 1.5}')
    client = FakeS3Client()
    store: Any = ArtifactStore(experiment_name="exp1", client=client, enabled=True)

    uri = store.upload_file(metrics, "eval/metrics.json")

    expected_key = "ml-data/artifacts/exp1/eval/metrics.json"
    assert uri == f"r2://{expected_key}"
    assert client.objects[expected_key] == b'{"si_sdr": 1.5}'


def test_upload_file_disabled_and_failure_are_noops(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_bytes(b"{}")

    disabled: Any = ArtifactStore(experiment_name="exp1", client=FakeS3Client(), enabled=False)
    assert disabled.upload_file(metrics, "eval/metrics.json") is None

    failing: Any = ArtifactStore(
        experiment_name="exp1",
        client=FakeS3Client(fail_keys={"ml-data/artifacts/exp1/eval/metrics.json"}),
        enabled=True,
    )
    assert failing.upload_file(metrics, "eval/metrics.json") is None


# ─── upload_val_samples ─────────────────────────────────────────────────────


def test_upload_val_samples_writes_audio_figures_and_manifest():
    client = FakeS3Client()
    store = ArtifactStore(experiment_name="exp1", client=client, enabled=True)

    wav = (np.linspace(-1.0, 1.0, 1600, dtype=np.float32), 16000)
    samples = [
        ValSample(
            sample_id="sample_000",
            input_snr=-12.5,
            audio={"mixture": wav, "target": wav, "output": wav},
            figures={"rps_overlay": b"\x89PNGfakebytes"},
            metrics={"mse": 0.05, "r2": 0.8},
        ),
        ValSample(sample_id="sample_001", input_snr=None, audio={"mixture": wav}),
    ]

    manifest_uri = store.upload_val_samples(7, samples)

    root = "ml-data/artifacts/exp1/val_samples/epoch_7"
    assert manifest_uri == f"r2://{root}/manifest.json"

    # Audio + figure keys present with the expected naming convention.
    assert f"{root}/sample_000__mixture.wav" in client.objects
    assert f"{root}/sample_000__target.wav" in client.objects
    assert f"{root}/sample_000__output.wav" in client.objects
    assert f"{root}/sample_000__rps_overlay.png" in client.objects
    assert client.objects[f"{root}/sample_000__rps_overlay.png"] == b"\x89PNGfakebytes"
    assert f"{root}/sample_001__mixture.wav" in client.objects

    # The wav bytes are a real, readable WAV file at the right sample rate.
    data, sr = sf.read(io.BytesIO(client.objects[f"{root}/sample_000__mixture.wav"]))
    assert sr == 16000
    assert len(data) == 1600

    manifest = json.loads(client.objects[f"{root}/manifest.json"])
    assert manifest["epoch"] == 7
    ids = {row["sample_id"]: row for row in manifest["samples"]}
    assert ids["sample_000"]["input_snr"] == -12.5
    assert ids["sample_000"]["metrics"] == {"mse": 0.05, "r2": 0.8}
    assert ids["sample_000"]["keys"]["mixture"] == f"r2://{root}/sample_000__mixture.wav"
    assert ids["sample_000"]["keys"]["rps_overlay"] == f"r2://{root}/sample_000__rps_overlay.png"
    assert ids["sample_001"]["input_snr"] is None


def test_upload_val_samples_empty_list_returns_none():
    client = FakeS3Client()
    store = ArtifactStore(experiment_name="exp1", client=client, enabled=True)
    assert store.upload_val_samples(0, []) is None
    assert client.objects == {}


def test_upload_val_samples_disabled_is_noop():
    client = FakeS3Client()
    store = ArtifactStore(experiment_name="exp1", client=client, enabled=False)
    wav = (np.zeros(100, dtype=np.float32), 16000)
    samples = [ValSample(sample_id="s0", audio={"mixture": wav})]

    assert store.upload_val_samples(0, samples) is None
    assert client.objects == {}


def test_upload_val_samples_exception_does_not_propagate():
    client = FakeS3Client(fail_keys={"ml-data/artifacts/exp1/val_samples/epoch_0/s0__mixture.wav"})
    store = ArtifactStore(experiment_name="exp1", client=client, enabled=True)
    wav = (np.zeros(100, dtype=np.float32), 16000)
    samples = [ValSample(sample_id="s0", audio={"mixture": wav})]

    result = store.upload_val_samples(0, samples)
    assert result is None


# ─── missing-env / no-client no-op path ─────────────────────────────────────


def test_load_r2_env_returns_none_when_vars_missing(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    for var in R2_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    assert _load_r2_env() is None


def test_load_r2_env_returns_values_when_present(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("R2_ACCOUNT_ID", "acct")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    env = _load_r2_env()
    assert env == {
        "R2_ACCOUNT_ID": "acct",
        "AWS_ACCESS_KEY_ID": "key",
        "AWS_SECRET_ACCESS_KEY": "secret",
    }


def test_store_without_client_or_env_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    for var in R2_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"x")
    store = ArtifactStore(experiment_name="exp1", enabled=True)  # no client injected

    assert store.active is False
    assert store.upload_checkpoint(ckpt) is None


# ─── results mirror: publish_results / download_tree ────────────────────────


def test_publish_results_mirrors_tree_without_weights_and_pulls_back(tmp_path):
    run = tmp_path / "results" / "exp1"
    files = {
        "eval/metrics.json": b'{"si_sdr": 1.5}',
        "checkpoints/validation_history.jsonl": b"{}\n",
        "a.csv": b"x\n1\n",
        "last.ckpt": b"weights",
        "wandb/run-1/run.wandb": b"wandb",
    }
    for rel, body in files.items():
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_bytes(body)
    client = FakeS3Client()

    uris = publish_results(run, results_root=tmp_path / "results", client=client)

    published = {"eval/metrics.json", "checkpoints/validation_history.jsonl", "a.csv"}
    assert sorted(uris) == sorted(f"r2://ml-data/artifacts/exp1/{rel}" for rel in published)
    assert publish_results(tmp_path / "elsewhere.csv", results_root=run, client=client) == []

    store = ArtifactStore(experiment_name="exp1", client=client)
    store.upload_checkpoint(run / "last.ckpt")
    pulled = tmp_path / "pulled"
    got = store.download_tree("", pulled)
    assert {p.relative_to(pulled).as_posix() for p in got} == published
    assert all((pulled / rel).read_bytes() == files[rel] for rel in published)
    assert store.download_tree("", pulled) == []  # size + mtime match: nothing refetched
    assert store.download_tree("checkpoints/last.ckpt", pulled / "w.ckpt", weights=True) == [
        pulled / "w.ckpt"
    ]
