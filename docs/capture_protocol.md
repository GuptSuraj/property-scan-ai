# Capture protocol

## Photos

- Take 4–8 photos per room when possible, from different positions or corners.
- Keep neighboring views overlapping and capture every wall.
- Include floor–wall and ceiling–wall boundaries.
- Capture doorways and one view through each doorway into the adjacent room.
- Hold the phone steady; avoid blur, extreme darkness, mirrors, and glass where possible.

Store one room per folder. The folder name becomes the room label.

## Video

- Walk slowly with a stable phone and turn gradually.
- Keep walls, floor, and ceiling boundaries visible.
- Record doorway transitions and revisit an earlier area when practical to form a useful loop.
- Avoid rapid motion, long views of blank walls, darkness, and moving people.

## LiDAR / RGB-D

The supported stock capture tool is **Record3D** on a LiDAR-capable iPhone or
iPad. In Record3D, start an RGB-D recording, walk slowly around one room, keep
the device stable, show every wall plus floor/ceiling boundaries, and revisit
the starting area when practical. Stop the recording and export/share the
recording in Record3D's native `.r3d` format. Copy that single file to the Mac,
then upload it in the UI or run:

```bash
python run.py --tier lidar --input ./inputs/room_scan.r3d --drift-correction on
```

Do not extract, rename, or reorder files inside the archive. The converter reads
Record3D's registered RGB, metric depth, intrinsics, and camera-to-world poses,
then explicitly converts OpenGL/ARKit axes to the application's Z-up convention.

For another capture tool, export the canonical folder described in
`docs/lidar_pipeline.md`: an explicit manifest, synchronized and registered
RGB/depth frames, pinhole intrinsics, and rigid poses. Declare depth units and
pose convention. Frame IDs in all files must match. Unregistered RGB/depth
streams are not accepted because the application does not guess alignment.
