# Device matrix

| Tier | Capture requirement | Expected processing | Accuracy status |
| --- | --- | --- | --- |
| Photo | Standard modern smartphone camera; 2–8 overlapping images per room | COLMAP SfM + local metric depth | To be filled from benchmark |
| Video | Standard modern smartphone camera; MP4 or MOV walkthrough | Sequential COLMAP + local metric depth | To be filled from benchmark |
| LiDAR | LiDAR/RGB-D capable device with canonical registered export | Metric RGB-D fusion and optional drift correction | To be filled from benchmark |

All processing targets Python 3.11 on Apple Silicon and uses MPS or CPU. No CUDA or cloud API is required.
