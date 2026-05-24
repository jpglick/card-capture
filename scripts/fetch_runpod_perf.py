import json, httpx
from pathlib import Path

def fetch_perf():
    cfg = json.loads(Path('card_capture_config.json').read_text())
    hdr = {'Authorization': f'Bearer {cfg["runpod_api_key"]}'}
    ep = cfg['runpod_endpoint_id']
    
    print(f"Fetching recent requests for endpoint {ep}...")
    try:
        r = httpx.get(f'https://api.runpod.ai/v2/{ep}/requests', headers=hdr)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Failed to fetch requests: {e}")
        if 'r' in locals():
            print(r.text)
        return

    requests = data.get('requests', [])
    if not requests:
        print("No recent requests found.")
        return

    for req in requests[:3]:
        job_id = req['id']
        print(f"\n--- Job {job_id} ---")
        try:
            status_r = httpx.get(f'https://api.runpod.ai/v2/{ep}/status/{job_id}', headers=hdr)
            status_r.raise_for_status()
            job_data = status_r.json()
            status = job_data.get('status', 'UNKNOWN')
            print(f"Status: {status}")
            
            output = job_data.get('output', {})
            if output:
                timings = output.get('timings', {})
                print(f"Timings: {json.dumps(timings, indent=2)}")
                
                resource_stats = output.get('resource_stats', {})
                print(f"Resource Stats: {json.dumps(resource_stats, indent=2)}")
                
                diagnostics = output.get('diagnostics', {})
                if 'events' in diagnostics:
                    print(f"Events summary: {diagnostics['events']}")
                if 'detect_telemetry' in diagnostics:
                    print(f"Detect Telemetry: {json.dumps(diagnostics['detect_telemetry'], indent=2)}")
            elif status == 'FAILED':
                error = job_data.get('error')
                if error:
                    print(f"Error: {error}")
        except Exception as e:
            print(f"Failed to fetch job {job_id}: {e}")

if __name__ == '__main__':
    fetch_perf()
