import asyncio
import json
from pathlib import Path
from app.services.event_bus import EventBus
from app.services.runpod_runner import RunPodRunner

async def main():
    cfg = json.loads(Path('card_capture_config.json').read_text())
    bus = EventBus()
    db_path = Path('card_capture_output/cards.sqlite')
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            from migrations.run_migrations import apply_migrations
            apply_migrations(db_path)
            
    runner = RunPodRunner(
        bus=bus,
        db_path=db_path,
        output_base=Path('card_capture_output'),
        api_key=cfg['runpod_api_key'],
        endpoint_id=cfg['runpod_endpoint_id'],
        r2_account_id=cfg['r2_account_id'],
        r2_bucket=cfg['r2_bucket'],
        r2_access_key_id=cfg['r2_access_key_id'],
        r2_secret_access_key=cfg['r2_secret_access_key'],
    )
    
    video = 'golden_set/videos/IMG_5872/video.mp4'
    if not Path(video).exists():
        print(f"Video {video} not found")
        return
        
    await runner.run_async(
        run_id='test_run_1',
        video=video,
        output_dir='card_capture_output/test_run_1',
        db=str(db_path),
        config_preset='balanced',
        video_id=1
    )

if __name__ == '__main__':
    asyncio.run(main())
