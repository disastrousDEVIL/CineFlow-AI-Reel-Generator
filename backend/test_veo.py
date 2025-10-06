import os
import sys
import time
import json
import base64
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import requests

# Import the project's auth initializer and defaults
if __package__ is None or __package__ == "":
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from backend.characters import _init_vertex_ai_if_needed, DEFAULT_PROJECT_ID, DEFAULT_LOCATION  # type: ignore
else:
    from .characters import _init_vertex_ai_if_needed, DEFAULT_PROJECT_ID, DEFAULT_LOCATION

try:
    import google.auth
    from google.auth.transport.requests import Request as GoogleAuthRequest
    import vertexai
    from vertexai.preview import vision_models
except Exception:
    google = None  # type: ignore
    GoogleAuthRequest = None  # type: ignore
    vertexai = None  # type: ignore
    vision_models = None  # type: ignore


def _validate_authentication() -> Tuple[bool, str]:
    """
    Validate that GCP authentication is properly configured.
    Returns (success: bool, message: str)
    """
    try:
        # Initialize Vertex AI (sets up credentials)
        _init_vertex_ai_if_needed()
        
        # Check if GOOGLE_APPLICATION_CREDENTIALS is set
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path:
            if not os.path.exists(creds_path):
                return False, f"[FAIL] GOOGLE_APPLICATION_CREDENTIALS points to non-existent file: {creds_path}"
            print(f"[PASS] Using credentials from: {creds_path}")
        else:
            print("[WARN] GOOGLE_APPLICATION_CREDENTIALS not set, using default credentials")
        
        # Try to get an access token
        if google is not None and GoogleAuthRequest is not None:
            try:
                creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])  # type: ignore
                print(f"[INFO] Credentials type: {type(creds).__name__}")
                print(f"[INFO] Project from credentials: {project}")
                
                # Refresh if needed
                if creds and hasattr(creds, 'expired') and creds.expired and hasattr(creds, 'refresh_token') and creds.refresh_token:
                    print("[INFO] Credentials expired, refreshing...")
                    creds.refresh(GoogleAuthRequest())  # type: ignore
                
                # For service account credentials, we need to refresh to get a token
                if creds and not hasattr(creds, 'token') or not creds.token:
                    print("[INFO] No token present, refreshing to obtain token...")
                    creds.refresh(GoogleAuthRequest())  # type: ignore
                
                if creds and hasattr(creds, 'token') and creds.token:
                    print(f"[PASS] Successfully authenticated with project: {project or DEFAULT_PROJECT_ID}")
                    print(f"[PASS] Access token obtained (first 20 chars): {creds.token[:20]}...")
                    return True, "Authentication successful"
                else:
                    return False, "[FAIL] Failed to obtain access token after refresh"
            except Exception as auth_error:
                return False, f"[FAIL] Authentication error: {str(auth_error)}"
        else:
            return False, "[FAIL] google.auth library not available"
            
    except Exception as e:
        return False, f"[FAIL] Authentication validation failed: {str(e)}"


def _get_access_token() -> str:
    """Get OAuth token using the same auth flow as characters.py"""
    # Ensure Vertex AI is initialized (sets GOOGLE_APPLICATION_CREDENTIALS if needed)
    _init_vertex_ai_if_needed()
    
    # Now use ADC (which picks up the configured credentials)
    if google is None or GoogleAuthRequest is None:
        raise RuntimeError("google.auth library not available")
    
    try:
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])  # type: ignore
        
        # Service account credentials need to be refreshed to get a token
        if creds and (not hasattr(creds, 'token') or not creds.token):
            creds.refresh(GoogleAuthRequest())  # type: ignore
        
        # Also refresh if expired
        if creds and hasattr(creds, 'expired') and creds.expired and hasattr(creds, 'refresh_token') and creds.refresh_token:
            creds.refresh(GoogleAuthRequest())  # type: ignore
        
        if creds and hasattr(creds, 'token') and creds.token:
            return creds.token
        
        raise RuntimeError("Failed to obtain access token")
    except Exception as e:
        raise RuntimeError(f"Failed to get token via ADC: {e}")


def _start_job(project_id: str, location: str, model: str, prompt: str, image_b64: Optional[str], duration: int, aspect_ratio: str, storage_uri: Optional[str] = None) -> Dict[str, Any]:
    """
    Start a Veo video generation job.
    
    Args:
        project_id: GCP project ID
        location: GCP location (e.g., 'us-central1')
        model: Model ID (e.g., 'veo-3.0-generate-001')
        prompt: Text prompt for video generation
        image_b64: Optional base64-encoded image for image-to-video
        duration: Video duration in seconds (4, 6, or 8 for Veo 3)
        aspect_ratio: Video aspect ratio ('16:9' or '9:16')
        storage_uri: Optional GCS bucket URI (e.g., 'gs://bucket/path/')
    
    Returns:
        Operation response dict with 'name' field
    """
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/"
        f"publishers/google/models/{model}:predictLongRunning"
    )
    headers = {"Authorization": f"Bearer {_get_access_token()}", "Content-Type": "application/json"}
    
    # Build instance
    instance: Dict[str, Any] = {"prompt": prompt}
    if image_b64:
        instance["image"] = {
            "bytesBase64Encoded": image_b64,
            "mimeType": "image/png"
        }
    
    # Build parameters according to documentation
    parameters: Dict[str, Any] = {
        "durationSeconds": duration,
        "aspectRatio": aspect_ratio,
        "sampleCount": 1
    }
    
    # Add storage URI if provided
    if storage_uri:
        parameters["storageUri"] = storage_uri
    
    payload = {
        "instances": [instance],
        "parameters": parameters
    }
    
    print(f"[DEBUG] Request payload: instances[0].keys={list(instance.keys())}, parameters.keys={list(parameters.keys())}")
    
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    data = r.json()
    print(f"[DEBUG] Response keys: {list(data.keys())}")
    
    # Check if operation is already done (some models return immediately)
    if data.get("done"):
        print("[INFO] Operation completed immediately!")
        return data
    
    # Store full response for polling attempts
    return data


def _poll_operation(project_id: str, location: str, model: str, op_name: str, timeout_sec: int = 1800, interval: float = 10.0) -> Dict[str, Any]:
    """
    Poll Veo operation using the fetchPredictOperation endpoint.
    According to Vertex AI documentation, Veo operations must be polled using this special endpoint.
    
    Reference: https://cloud.google.com/vertex-ai/docs/generative-ai/video/generate-videos
    """
    # Extract operation ID from the full operation name
    # Format: projects/PROJECT_ID/locations/LOCATION/publishers/google/models/MODEL_ID/operations/OPERATION_ID
    operation_id = op_name.split("/")[-1]
    
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/"
        f"publishers/google/models/{model}:fetchPredictOperation"
    )
    
    headers = {"Authorization": f"Bearer {_get_access_token()}", "Content-Type": "application/json"}
    payload = {"operationName": op_name}
    
    print(f"[INFO] Polling operation (typically takes 2-5 minutes)...")
    print(f"[INFO] Operation ID: {operation_id}")
    
    start_time = time.time()
    last_status_time = 0
    
    end = time.time() + timeout_sec
    while time.time() < end:
        try:
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            # Check if operation is done
            if data.get("done"):
                print("[PASS] Operation completed successfully!")
                return data
            
            # Print status update every 30 seconds
            elapsed = int(time.time() - start_time)
            if time.time() - last_status_time >= 30:
                print(f"[INFO] Still generating... ({elapsed}s elapsed)")
                last_status_time = time.time()
                
        except requests.exceptions.HTTPError as e:
            print(f"[WARN] HTTP error while polling: {e}")
            if e.response.status_code >= 500:
                # Server error, retry after delay
                time.sleep(interval * 2)
                continue
            else:
                # Client error, don't retry
                raise
        except Exception as e:
            print(f"[WARN] Error while polling: {e}")
        
        time.sleep(interval)
    
    raise TimeoutError(f"Operation did not complete within {timeout_sec}s ({timeout_sec/60:.1f} minutes)")


def _extract_video(op: Dict[str, Any]) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Extract video from operation response.
    Response format according to Vertex AI docs:
    {
        "done": true,
        "response": {
            "@type": "type.googleapis.com/cloud.ai.large_models.vision.GenerateVideoResponse",
            "raiMediaFilteredCount": 0,
            "videos": [
                {
                    "gcsUri": "gs://...",
                    "mimeType": "video/mp4"
                }
            ]
        }
    }
    """
    if "error" in op:
        error_msg = op["error"]
        raise RuntimeError(f"Operation failed: {error_msg}")
    
    resp = op.get("response", {})
    
    # Check for RAI filtering
    rai_filtered = resp.get("raiMediaFilteredCount", 0)
    if rai_filtered > 0:
        reasons = resp.get("raiMediaFilteredReasons", [])
        print(f"[WARN] {rai_filtered} video(s) filtered due to safety policies: {reasons}")
    
    # Get videos array
    videos = resp.get("videos", [])
    if not videos:
        print("[WARN] No videos in response")
        return None, None
    
    # Get first video
    video = videos[0]
    
    # Check for base64 encoded video
    if video.get("bytesBase64Encoded"):
        try:
            video_bytes = base64.b64decode(video["bytesBase64Encoded"])
            print(f"[INFO] Received video as base64 ({len(video_bytes)} bytes)")
            return video_bytes, None
        except Exception as e:
            print(f"[WARN] Failed to decode base64 video: {e}")
    
    # Check for GCS URI
    if video.get("gcsUri"):
        gcs_uri = video["gcsUri"]
        print(f"[INFO] Video stored at: {gcs_uri}")
        return None, gcs_uri
    
    print("[WARN] Video object has no bytesBase64Encoded or gcsUri field")
    return None, None


def validate_only():
    """Run authentication validation only"""
    print("\n" + "="*60)
    print("[AUTH] GCP AUTHENTICATION VALIDATION")
    print("="*60 + "\n")
    
    print(f"[INFO] Project ID: {DEFAULT_PROJECT_ID}")
    print(f"[INFO] Location: {DEFAULT_LOCATION}\n")
    
    success, message = _validate_authentication()
    
    print("\n" + "="*60)
    if success:
        print("[PASS] VALIDATION PASSED")
        print(message)
    else:
        print("[FAIL] VALIDATION FAILED")
        print(message)
    print("="*60 + "\n")
    
    return 0 if success else 1


def main():
    # Check if user wants validation only
    if os.getenv("VALIDATE_ONLY", "").lower() in ("1", "true", "yes"):
        return validate_only()
    
    # Validate authentication before proceeding
    print("\n[AUTH] Validating GCP authentication...")
    success, message = _validate_authentication()
    if not success:
        print(f"\n[FAIL] Authentication validation failed: {message}")
        print("[TIP] Run with VALIDATE_ONLY=true to see detailed validation info")
        return 1
    print("")  # Add spacing
    
    project_id = DEFAULT_PROJECT_ID
    location = DEFAULT_LOCATION
    model = os.getenv("VIDEO_MODEL", "veo-3.0-generate-001")
    duration = int(os.getenv("DURATION", "6"))
    aspect = os.getenv("ASPECT_RATIO", "9:16")
    prompt = os.getenv("PROMPT", "Cinematic camera motion over a city at night, realistic lighting.")
    frame_path = os.getenv("FRAME_PATH", "")
    out_path = os.getenv("OUT", "videos/test_veo.mp4")

    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)

    image_b64 = None
    if frame_path and Path(frame_path).exists():
        with open(frame_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

    print(f"[VIDEO] Starting Veo LRO: model={model}, duration={duration}, aspect={aspect}")
    print(f"[VIDEO] Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}\n")
    
    try:
        op_data = _start_job(project_id, location, model, prompt, image_b64, duration, aspect)
        op_name = op_data.get("name", "unknown")
        print(f"[PASS] Operation started: {op_name}")
        
        # Check if already done
        if op_data.get("done"):
            print("[INFO] Operation completed immediately!")
            op = op_data
        else:
            print("[VIDEO] Waiting for completion (this may take several minutes)...\n")
            op = _poll_operation(project_id, location, model, op_name)
        
        video_bytes, gcs_uri = _extract_video(op)
        
        if video_bytes:
            with open(out_path, "wb") as f:
                f.write(video_bytes)
            print(f"[PASS] Video saved to: {out_path}")
        elif gcs_uri:
            print(f"[PASS] Video generated at GCS: {gcs_uri}")
        else:
            print("[WARN] No video payload in operation response.")
            return 1
            
        return 0
    except Exception as e:
        print(f"\n[FAIL] Error during video generation: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        sys.exit(exit_code)
