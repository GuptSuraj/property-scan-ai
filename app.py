"""Minimal local Streamlit front end for the unified scanner."""
from pathlib import Path
import json, tempfile, zipfile


def safe_extract_zip(archive: Path, destination: Path) -> Path:
    destination = destination.resolve(); destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination/member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
        handle.extractall(destination)
    children = [p for p in destination.iterdir() if p.name != "__MACOSX"]
    return children[0] if len(children) == 1 and children[0].is_dir() else destination


def result_summary(payload: dict) -> dict:
    prop = payload.get("property", {}); rooms = prop.get("rooms", [])
    area = prop.get("total_floor_area") or {}
    return {"Rooms detected": len(rooms), "Total floor area": area.get("value"),
        "Openings detected": len(prop.get("openings", [])), "Damage regions": len(payload.get("damages", [])),
        "Warnings": len(payload.get("warnings", []))}


def main() -> None:
    import streamlit as st
    from config.settings import load_settings
    from property_scanner.inputs import select_adapter
    from property_scanner.pipeline.processor import PropertyScanPipeline
    from property_scanner.reconstruction.lidar.models import LidarConfig
    from property_scanner.reconstruction.video.models import VideoConfig
    from property_scanner.reconstruction.photo.models import PhotoConfig
    st.set_page_config(page_title="AI Property Scanner", layout="wide"); st.title("AI Property Scanner")
    tier_label = st.selectbox("Input tier", ("Photos", "Video", "LiDAR")); tier = {"Photos": "photo", "Video": "video", "LiDAR": "lidar"}[tier_label]
    local = st.text_input("Local input path (development)")
    types = ["zip"] if tier != "video" else ["mp4", "mov"]
    uploaded = st.file_uploader("Or upload a capture", type=types)
    skip_damage = st.checkbox("Skip damage analysis", value=False)
    if st.button("Process Property", type="primary"):
        if not local and uploaded is None: st.error("Choose an upload or enter a local path."); return
        try:
            with tempfile.TemporaryDirectory(prefix="property-scanner-") as temp:
                if local: source = Path(local).expanduser().resolve()
                else:
                    uploaded_path = Path(temp)/uploaded.name; uploaded_path.write_bytes(uploaded.getbuffer())
                    source = uploaded_path if tier == "video" else safe_extract_zip(uploaded_path, Path(temp)/"capture")
                with st.status("Processing property", expanded=True) as status:
                    settings = load_settings(); pipeline = PropertyScanPipeline(settings)
                    st.write("Validating capture"); prepared = pipeline.prepare(select_adapter(tier, source))
                    st.write("Reconstructing and running shared analysis")
                    configs = {"photo_config": PhotoConfig()} if tier == "photo" else {"video_config": VideoConfig()} if tier == "video" else {"lidar_config": LidarConfig()}
                    result = pipeline.process(prepared, skip_damage=skip_damage, **configs); status.update(label="Processing complete", state="complete")
                output = prepared.output_dir; payload = json.loads((output/"result.json").read_text())
                st.subheader("Floor Plan")
                if (output/"floorplan.png").is_file(): st.image(str(output/"floorplan.png"))
                st.subheader("Summary"); st.json(result_summary(payload))
                for room in payload["property"]["rooms"]:
                    with st.expander(room.get("name") or room["room_id"]): st.json(room)
                for filename, mime in (("result.json", "application/json"), ("floorplan.png", "image/png"), ("floorplan.svg", "image/svg+xml"), ("measurements.csv", "text/csv")):
                    path = output/filename
                    if path.is_file(): st.download_button(f"Download {filename}", path.read_bytes(), filename, mime)
        except Exception as exc:
            st.error(f"Processing failed: {exc}")


if __name__ == "__main__": main()
