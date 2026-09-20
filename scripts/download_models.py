"""Explicit local metric-depth and opening-segmentation downloads; weights stay ignored."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',action='store_true',help='Download pinned Depth Anything V2 Small metric indoor weights')
    parser.add_argument('--photo',action='store_true',help='Download the same shared metric-depth weights for photo mode')
    parser.add_argument('--openings',action='store_true',help='Download pinned SegFormer-B0 ADE20K opening segmentation weights')
    parser.add_argument('--damage',action='store_true',help='Download the YOLOE-11s segmentation checkpoint for damage prompts')
    args=parser.parse_args()
    if not (args.video or args.photo or args.openings or args.damage):
        print('No download requested. Use --photo, --video, --openings, or --damage.')
        return 0
    from huggingface_hub import snapshot_download
    from config.settings import load_settings
    settings=load_settings()
    if args.video or args.photo:
        from property_scanner.reconstruction.video.models import DEPTH_MODEL, DEPTH_REVISION
        from property_scanner.reconstruction.video.depth import CACHE_NAME
        target=settings.model_dir/CACHE_NAME
        snapshot_download(DEPTH_MODEL,revision=DEPTH_REVISION,local_dir=target,
                          allow_patterns=['*.json','*.safetensors'],etag_timeout=30)
        (target/'provenance.json').write_text(__import__('json').dumps({'model':DEPTH_MODEL,'revision':DEPTH_REVISION},indent=2))
        print(f'Metric indoor model cached: {target}')
    if args.openings:
        from property_scanner.openings.models import SEMANTIC_MODEL, SEMANTIC_REVISION, SEMANTIC_CACHE_NAME
        target=settings.model_dir/SEMANTIC_CACHE_NAME
        snapshot_download(SEMANTIC_MODEL,revision=SEMANTIC_REVISION,local_dir=target,
                          allow_patterns=['*.json','*.safetensors'],etag_timeout=30)
        (target/'provenance.json').write_text(__import__('json').dumps(
            {'model':SEMANTIC_MODEL,'revision':SEMANTIC_REVISION},indent=2))
        print(f'Opening segmentation model cached: {target}')
    if args.damage:
        try:
            from ultralytics.utils.downloads import attempt_download_asset
            from property_scanner.damage.models import DAMAGE_MODEL_FILE
            import shutil
            source = Path(attempt_download_asset(DAMAGE_MODEL_FILE))
            target = settings.model_dir/DAMAGE_MODEL_FILE
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != target.resolve(): shutil.copy2(source, target)
            print(f'Damage segmentation model cached: {target}')
            print('Prompt-free YOLOE avoids a separate text-encoder download at processing time.')
        except ImportError as exc:
            print("Damage download requires the damage extra: pip install -e '.[damage]'", file=sys.stderr)
            return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
