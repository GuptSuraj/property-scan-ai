"""Check real preparation behavior and enforce honest processing boundaries."""

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from config.settings import Settings, load_settings
from property_scanner.core.exceptions import (
    ConfigurationError, InvalidInputError, ProcessingError, UnsupportedTierError,
)
from property_scanner.core.paths import create_capture_output
from property_scanner.inputs import select_adapter
from property_scanner.inputs.lidar import LidarInputAdapter
from property_scanner.inputs.photo import PhotoInputAdapter
from property_scanner.inputs.video import VideoInputAdapter
from property_scanner.pipeline.context import ScanContext
from property_scanner.pipeline.processor import PropertyScanPipeline
from property_scanner.schemas.common import NormalizedCapture, PreparationResult

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("APP_ENV", "LOG_LEVEL", "MODEL_DIR", "OUTPUT_DIR", "INPUT_DIR"):
        monkeypatch.delenv(name, raising=False)


def make_source(root: Path, tier: str) -> Path:
    """Synthetic path fixtures are never presented as decoded media."""
    if tier == "video":
        path = root / "walkthrough.MP4"
        path.write_bytes(b"path-validation fixture, not a real video")
    else:
        path = root / tier
        path.mkdir()
        if tier == "photo":
            (path / "one.jpg").write_bytes(b"path fixture")
            (path / "two.PNG").write_bytes(b"path fixture")
        else:
            (path / "metadata.json").write_text("{}", encoding="utf-8")
    return path


def test_package_import() -> None:
    assert importlib.import_module("property_scanner").__version__ == "0.1.0"


@pytest.mark.parametrize("tier,expected", [
    ("photo", PhotoInputAdapter), ("video", VideoInputAdapter), ("lidar", LidarInputAdapter),
])
def test_adapter_selection_and_normalization(tmp_path: Path, tier: str, expected: type) -> None:
    source = make_source(tmp_path, tier)
    adapter = select_adapter(tier, source)
    assert isinstance(adapter, expected)
    adapter.validate()
    capture = adapter.prepare()
    assert capture.tier == tier
    assert capture.source_path == source.resolve()
    assert capture.prepared_files
    assert capture.created_at.utcoffset().total_seconds() == 0
    assert capture.metadata["media_decoded"] is False
    assert NormalizedCapture.model_validate_json(capture.model_dump_json()) == capture


def test_invalid_tier() -> None:
    with pytest.raises(UnsupportedTierError, match="Unsupported tier"):
        select_adapter("satellite", Path("missing"))


@pytest.mark.parametrize("tier", ["photo", "video", "lidar"])
def test_missing_path(tier: str) -> None:
    with pytest.raises(InvalidInputError, match="does not exist"):
        select_adapter(tier, Path("missing")).prepare()


@pytest.mark.parametrize("tier", ["photo", "lidar"])
def test_directory_required(tmp_path: Path, tier: str) -> None:
    source = tmp_path / "file.txt"
    source.touch()
    with pytest.raises(InvalidInputError, match="directory"):
        select_adapter(tier, source).prepare()


@pytest.mark.parametrize("count", [0, 1, 9])
def test_photo_count_limits(tmp_path: Path, count: int) -> None:
    for index in range(count):
        (tmp_path / f"{index}.jpg").touch()
    with pytest.raises(InvalidInputError, match="2–8"):
        PhotoInputAdapter(tmp_path).prepare()


def test_video_extension_and_directory(tmp_path: Path) -> None:
    source = tmp_path / "capture.txt"
    source.touch()
    with pytest.raises(InvalidInputError, match="extension"):
        VideoInputAdapter(source).prepare()
    with pytest.raises(InvalidInputError, match="must be a file"):
        VideoInputAdapter(tmp_path).prepare()


def test_lidar_rejects_placeholder_only(tmp_path: Path) -> None:
    (tmp_path / ".gitkeep").touch()
    with pytest.raises(InvalidInputError, match="at least one"):
        LidarInputAdapter(tmp_path).prepare()


def test_output_directory_and_no_fake_results(tmp_path: Path) -> None:
    source = make_source(tmp_path, "photo")
    pipeline = PropertyScanPipeline(Settings())
    prepared = pipeline.prepare(PhotoInputAdapter(source))
    assert prepared.output_dir == tmp_path / "outputs" / str(prepared.capture.capture_id)
    assert prepared.output_dir.is_dir()
    assert list(prepared.output_dir.iterdir()) == []
    assert prepared.status == "prepared_not_processed"
    assert PreparationResult.model_validate_json(prepared.model_dump_json()) == prepared
    with pytest.raises(NotImplementedError, match="Scene reconstruction"):
        pipeline.process(prepared)
    for stage in pipeline.stages:
        with pytest.raises(NotImplementedError, match="not implemented"):
            stage.run(ScanContext(prepared.capture, prepared.output_dir))
    with pytest.raises(ProcessingError, match="Cannot create output"):
        create_capture_output(pipeline.settings.output_dir, prepared.capture.capture_id)


def test_invalid_input_does_not_create_output(tmp_path: Path) -> None:
    pipeline = PropertyScanPipeline(Settings())
    with pytest.raises(InvalidInputError):
        pipeline.prepare(PhotoInputAdapter(tmp_path / "missing"))
    assert not pipeline.settings.output_dir.exists()


def test_settings_environment_and_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("APP_ENV=from-file\nLOG_LEVEL=warning\n", encoding="utf-8")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("OUTPUT_DIR", "custom-output")
    monkeypatch.setenv("MODEL_DIR", "custom-models")
    monkeypatch.setenv("INPUT_DIR", "custom-inputs")
    settings = load_settings()
    assert settings.app_env == "test"
    assert settings.log_level == "WARNING"
    assert settings.output_dir == tmp_path / "custom-output"
    assert settings.model_dir == tmp_path / "custom-models"
    assert settings.input_dir == tmp_path / "custom-inputs"
    assert not settings.output_dir.exists()


@pytest.mark.parametrize("name,value", [("LOG_LEVEL", "unknown"), ("OUTPUT_DIR", ""), ("APP_ENV", "")])
def test_invalid_settings(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError):
        load_settings()


def run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "run.py"), *args],
        cwd=tmp_path, env=os.environ.copy(), capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize("tier", ["photo", "video", "lidar"])
def test_cli_tiers(tmp_path: Path, tier: str) -> None:
    source = make_source(tmp_path, tier)
    result = run_cli(tmp_path, "--tier", tier, "--input", str(source))
    assert result.returncode == 0, result.stderr
    assert "prepared_not_processed" in result.stdout
    assert "not implemented" in result.stdout
    outputs = list((tmp_path / "outputs").iterdir())
    assert len(outputs) == 1
    assert outputs[0].is_dir()
    assert not list(outputs[0].iterdir())


@pytest.mark.parametrize("tier,message", [("photo", "does not exist"), ("invalid", "invalid choice")])
def test_cli_clean_errors(tmp_path: Path, tier: str, message: str) -> None:
    result = run_cli(tmp_path, "--tier", tier, "--input", "missing")
    assert result.returncode == 2
    assert message in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_configuration_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "unknown")
    result = run_cli(tmp_path, "--tier", "photo", "--input", "missing")
    assert result.returncode == 2
    assert "Invalid settings" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_output_creation_error(tmp_path: Path) -> None:
    source = make_source(tmp_path, "video")
    (tmp_path / "outputs").write_text("blocked", encoding="utf-8")
    result = run_cli(tmp_path, "--tier", "video", "--input", str(source))
    assert result.returncode == 2
    assert "directory" in result.stderr
    assert "Traceback" not in result.stderr
