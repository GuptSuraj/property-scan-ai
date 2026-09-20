"""Explicit optional video model download; no weights enter Git."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',action='store_true',help='Download pinned Depth Anything V2 Small metric indoor weights')
    args=parser.parse_args()
    if not args.video:
        print('No download requested. Foundation, geometry, rendering and LiDAR require no AI models. Use --video for metric depth.')
        return 0
    from huggingface_hub import snapshot_download
    from config.settings import load_settings
    from property_scanner.reconstruction.video.models import DEPTH_MODEL, DEPTH_REVISION
    from property_scanner.reconstruction.video.depth import CACHE_NAME
    target=load_settings().model_dir/CACHE_NAME
    snapshot_download(DEPTH_MODEL,revision=DEPTH_REVISION,local_dir=target,
                      allow_patterns=['*.json','*.safetensors'],etag_timeout=30)
    (target/'provenance.json').write_text(__import__('json').dumps({'model':DEPTH_MODEL,'revision':DEPTH_REVISION},indent=2))
    print(f'Metric indoor model cached: {target}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
