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

Export the canonical folder described in `docs/lidar_pipeline.md`: an explicit manifest, synchronized and registered RGB/depth frames, pinhole intrinsics, and rigid poses. Declare depth units and pose convention. Do not rename or reorder frames; IDs in all files must match. This repository does not assume an app-specific export format.
