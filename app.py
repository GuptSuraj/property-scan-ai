"""Local Streamlit front end for the unified property-scanning pipeline."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import zipfile

MAX_EXTRACTED_BYTES = 4 * 1024**3
MAX_ZIP_MEMBERS = 10_000


def safe_extract_zip(archive: Path, destination: Path) -> Path:
    """Extract a bounded ZIP without absolute paths or parent traversal."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        if len(members) > MAX_ZIP_MEMBERS:
            raise ValueError("ZIP contains too many files")
        if sum(member.file_size for member in members) > MAX_EXTRACTED_BYTES:
            raise ValueError("ZIP expands beyond the 4 GB safety limit")
        for member in members:
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
        handle.extractall(destination)
    children = [path for path in destination.iterdir() if path.name != "__MACOSX"]
    return children[0] if len(children) == 1 and children[0].is_dir() else destination


def result_summary(payload: dict) -> dict:
    prop = payload.get("property", {})
    area = prop.get("total_floor_area") or {}
    return {
        "Rooms detected": len(prop.get("rooms", [])),
        "Total floor area": area.get("value"),
        "Openings detected": len(prop.get("openings", [])),
        "Damage regions": len(payload.get("damages", [])),
        "Warnings": len(payload.get("warnings", [])),
    }


def artifact_inventory(output: Path) -> list[str]:
    if not output.is_dir():
        return []
    return sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())


def _measurement_rows(payload: dict) -> list[dict]:
    rows = []
    prop = payload.get("property", {})
    for room in prop.get("rooms", []):
        for wall in room.get("walls", []):
            if wall.get("length"):
                rows.append(_row(room["room_id"], wall["wall_id"], "wall length", wall["length"]))
        floor = room.get("floor") or {}
        ceiling = room.get("ceiling") or {}
        if floor.get("area"): rows.append(_row(room["room_id"], floor.get("surface_id"), "floor area", floor["area"]))
        if ceiling.get("height"): rows.append(_row(room["room_id"], ceiling.get("surface_id"), "ceiling height", ceiling["height"]))
    for opening in prop.get("openings", []):
        for name, field in (("opening width", "width"), ("opening height", "height"), ("sill height", "sill_height")):
            if opening.get(field): rows.append(_row(", ".join(opening.get("room_ids", [])), opening.get("opening_id"), name, opening[field]))
    for damage in payload.get("damages", []):
        if damage.get("metric_area"): rows.append(_row(damage.get("room_id"), damage.get("surface_id"), "damage area", damage["metric_area"]))
        if damage.get("metric_length"): rows.append(_row(damage.get("room_id"), damage.get("surface_id"), "damage length", damage["metric_length"]))
    return rows


def _row(room: str | None, surface: str | None, kind: str, value: dict) -> dict:
    interval = "Unavailable"
    if value.get("lower_bound") is not None and value.get("upper_bound") is not None:
        interval = f"{value['lower_bound']:.3f}–{value['upper_bound']:.3f} {value['unit']}"
    return {"Room": room or "—", "Surface": surface or "—", "Measurement": kind,
        "Value": f"{value['value']:.3f} {value['unit']}", "Interval": interval,
        "Confidence method": value.get("confidence_method", "unavailable")}


def _render_results(st, output: Path) -> None:
    result_path = output / "result.json"
    if not result_path.is_file():
        st.warning("No result.json was generated.")
        return
    payload = json.loads(result_path.read_text())
    summary = result_summary(payload)
    st.subheader("Summary")
    columns = st.columns(5)
    for column, (label, value) in zip(columns, summary.items(), strict=True):
        display = "Unavailable" if value is None else f"{value:.2f} m²" if label == "Total floor area" else value
        column.metric(label, display)

    errors = (payload.get("processing_info") or {}).get("errors", [])
    if errors:
        st.error("Processing completed with errors. Available reconstruction artifacts were retained.")
        for issue in errors: st.write(f"**{issue['code']}** — {issue['message']}")
    warnings = payload.get("warnings", [])
    if warnings:
        with st.expander(f"Warnings ({len(warnings)})", expanded=True):
            for warning in warnings: st.write(f"**{warning['code']}** — {warning['message']}")

    st.subheader("Floor plan")
    root_plan = output / "floorplan.png"
    if root_plan.is_file():
        st.image(str(root_plan), caption="Whole-property floor plan")
    else:
        room_plans = sorted((output / "photo").glob("*/floorplan.png"))
        if room_plans:
            st.info("A verified whole-property stitch was unavailable; showing room-local plans.")
            for plan in room_plans: st.image(str(plan), caption=plan.parent.name)
        else:
            st.warning("No floor plan was produced. Review processing errors and diagnostics below.")

    st.subheader("Measurements")
    rows = _measurement_rows(payload)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No metric measurements were resolved.")

    openings = payload.get("property", {}).get("openings", [])
    damages = payload.get("damages", [])
    scope = payload.get("scope_line_items", [])
    for room in payload.get("property", {}).get("rooms", []):
        with st.expander(room.get("name") or room["room_id"]):
            floor = room.get("floor") or {}; ceiling = room.get("ceiling") or {}
            st.write({"Room ID": room["room_id"],
                "Floor area": (floor.get("area") or {}).get("value"),
                "Ceiling height": (ceiling.get("height") or {}).get("value"),
                "Walls": len(room.get("walls", [])),
                "Openings": sum(room["room_id"] in item.get("room_ids", []) for item in openings),
                "Damage regions": sum(item.get("room_id") == room["room_id"] for item in damages)})

    if damages or scope:
        st.subheader("Damage and repair scope")
        if damages:
            st.dataframe([{"Damage": item["damage_type"], "Room": item.get("room_id"),
                "Surface": item.get("surface_id"),
                "Area (m²)": (item.get("metric_area") or {}).get("value"),
                "Length (m)": (item.get("metric_length") or {}).get("value"),
                "Views": len(item.get("source_frames", []))} for item in damages], hide_index=True, use_container_width=True)
        if scope:
            st.dataframe([{"Action": item["action"], "Description": item["description"],
                "Room": item.get("room_id"), "Surface": item.get("surface_id")} for item in scope],
                hide_index=True, use_container_width=True)

    st.subheader("Downloads")
    for filename, mime in (("result.json", "application/json"), ("floorplan.png", "image/png"),
                           ("floorplan.svg", "image/svg+xml"), ("measurements.csv", "text/csv")):
        path = output / filename
        if path.is_file(): st.download_button(f"Download {filename}", path.read_bytes(), filename, mime, key=f"download-{filename}")
    with st.expander("Generated artifacts"):
        st.code("\n".join(artifact_inventory(output)) or "No artifacts")


def main() -> None:
    import streamlit as st
    from config.settings import load_settings
    from property_scanner.inputs import select_adapter
    from property_scanner.pipeline.processor import PropertyScanPipeline
    from property_scanner.reconstruction.lidar.models import LidarConfig
    from property_scanner.reconstruction.video.models import VideoConfig
    from property_scanner.reconstruction.photo.models import PhotoConfig

    st.set_page_config(page_title="AI Property Scanner", layout="wide")
    st.title("AI Property Scanner")
    st.caption("Local Photo, Video, and canonical LiDAR/RGB-D reconstruction. No API key is required.")
    tier_label = st.selectbox("Input tier", ("Photos", "Video", "LiDAR"))
    tier = {"Photos": "photo", "Video": "video", "LiDAR": "lidar"}[tier_label]
    help_text = {"photo": "Property folder, or ZIP with one folder per room.",
        "video": "Full path to an MP4/MOV walkthrough, or upload it below.",
        "lidar": "Record3D .r3d, canonical RGB-D folder, or ZIP containing manifest.json."}[tier]
    local = st.text_input("Local input path (optional)", help=help_text, placeholder={
        "photo": "/Users/name/property/living_room/...", "video": "/Users/name/walkthrough.mov",
        "lidar": "/Users/name/lidar_capture"}[tier])
    types = ["zip"] if tier == "photo" else ["mp4", "mov"] if tier == "video" else ["zip", "r3d"]
    uploaded = st.file_uploader("Upload capture", type=types, help="Use either an upload or a local path; a local path takes precedence.")
    drift = st.selectbox("LiDAR drift correction", ("on", "off"), disabled=tier != "lidar")
    skip_damage = st.checkbox("Skip damage analysis", value=False,
        help="Use this for a faster geometry-only run. No damage objects will be invented.")

    prepared = None
    if st.button("Process Property", type="primary", use_container_width=True):
        if not local and uploaded is None:
            st.error("Choose an upload or enter a local path.")
        else:
            try:
                with tempfile.TemporaryDirectory(prefix="property-scanner-") as temp:
                    if local:
                        source = Path(local).expanduser().resolve()
                    else:
                        uploaded_path = Path(temp) / uploaded.name
                        uploaded_path.write_bytes(uploaded.getbuffer())
                        source = uploaded_path if tier == "video" or uploaded_path.suffix.lower() == ".r3d" else safe_extract_zip(uploaded_path, Path(temp) / "capture")
                    if tier == "lidar":
                        if source.suffix.lower() == ".r3d":
                            from property_scanner.reconstruction.lidar.record3d import convert_record3d
                            source = convert_record3d(source, Path(temp) / "canonical_record3d")
                        else:
                            from property_scanner.reconstruction.lidar.stray import is_stray_scanner_dir, convert_stray_scanner
                            if is_stray_scanner_dir(source):
                                source = convert_stray_scanner(source, Path(temp) / "canonical_stray")
                    progress = st.progress(0, text="[1/8] Validating capture")
                    settings = load_settings(); pipeline = PropertyScanPipeline(settings)
                    prepared = pipeline.prepare(select_adapter(tier, source)); progress.progress(15, text="[2/8] Reconstructing scene")
                    configs = {"photo_config": PhotoConfig()} if tier == "photo" else {
                        "video_config": VideoConfig()} if tier == "video" else {
                        "lidar_config": LidarConfig(drift_correction=drift)}
                    result = pipeline.process(prepared, skip_damage=skip_damage, **configs)
                    progress.progress(100, text="[8/8] Outputs complete")
                    st.session_state["last_result_dir"] = str(prepared.output_dir)
                    if result.processing_info and result.processing_info.errors:
                        st.warning("Processing retained partial outputs. Review the errors below.")
                    else:
                        st.success("Property processing completed.")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                if prepared is not None:
                    st.info(f"Retained output directory: {prepared.output_dir}")
                    with st.expander("Artifacts retained before failure"):
                        st.code("\n".join(artifact_inventory(prepared.output_dir)) or "No artifacts")

    last = st.session_state.get("last_result_dir")
    if last:
        output = Path(last)
        if output.is_dir(): _render_results(st, output)


if __name__ == "__main__":
    main()
