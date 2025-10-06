import os
import sys
import json
import base64
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

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
except Exception:
    google = None  # type: ignore
    GoogleAuthRequest = None  # type: ignore

import requests


def _get_access_token() -> str:
    """Get OAuth token using the same auth flow as test_veo.py"""
    _init_vertex_ai_if_needed()
    
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


def _start_video_job(
    project_id: str,
    location: str,
    model: str,
    prompt: str,
    image_path: Optional[str],
    duration: int,
    aspect_ratio: str,
    storage_uri: Optional[str] = None
) -> Dict[str, Any]:
    """
    Start a Veo video generation job with an input image.
    
    Args:
        project_id: GCP project ID
        location: GCP location (e.g., 'us-central1')
        model: Model ID (e.g., 'veo-3.0-generate-001')
        prompt: Text prompt for video generation
        image_path: Path to the character image
        duration: Video duration in seconds (4, 6, or 8 for Veo 3)
        aspect_ratio: Video aspect ratio ('16:9' or '9:16')
        storage_uri: Optional GCS bucket URI
    
    Returns:
        Operation response dict with 'name' field
    """
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/"
        f"publishers/google/models/{model}:predictLongRunning"
    )
    headers = {"Authorization": f"Bearer {_get_access_token()}", "Content-Type": "application/json"}
    
    # Build instance; optionally include image
    instance: Dict[str, Any] = {"prompt": prompt}
    if image_path:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        instance["image"] = {
            "bytesBase64Encoded": image_b64,
            "mimeType": "image/png"
        }
    
    # Build parameters according to documentation
    parameters: Dict[str, Any] = {
        "durationSeconds": duration,
        "aspectRatio": aspect_ratio,
        "sampleCount": 1,
        # Veo 3 requires generateAudio boolean; set to false by default
        "generateAudio": False
    }
    
    if storage_uri:
        parameters["storageUri"] = storage_uri
    
    payload = {
        "instances": [instance],
        "parameters": parameters
    }
    
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    data = r.json()
    
    return data


def _poll_operation(
    project_id: str,
    location: str,
    model: str,
    op_name: str,
    timeout_sec: int = 1800,
    interval: float = 10.0
) -> Dict[str, Any]:
    """
    Poll Veo operation using the fetchPredictOperation endpoint.
    Reference: https://cloud.google.com/vertex-ai/docs/generative-ai/video/generate-videos
    """
    operation_id = op_name.split("/")[-1]
    
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/"
        f"publishers/google/models/{model}:fetchPredictOperation"
    )
    
    headers = {"Authorization": f"Bearer {_get_access_token()}", "Content-Type": "application/json"}
    payload = {"operationName": op_name}
    
    print(f"   [POLL] Operation ID: {operation_id}")
    
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
                return data
            
            # Print status update every 30 seconds
            elapsed = int(time.time() - start_time)
            if time.time() - last_status_time >= 30:
                print(f"   [POLL] Still generating... ({elapsed}s elapsed)")
                last_status_time = time.time()
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code >= 500:
                # Server error, retry after delay
                time.sleep(interval * 2)
                continue
            else:
                raise
        except Exception as e:
            print(f"   [WARN] Polling error: {e}")
        
        time.sleep(interval)
    
    raise TimeoutError(f"Operation did not complete within {timeout_sec}s ({timeout_sec/60:.1f} minutes)")


def _extract_video_bytes(op: Dict[str, Any]) -> Optional[bytes]:
    """
    Extract video bytes from operation response.
    """
    if "error" in op:
        error_msg = op["error"]
        raise RuntimeError(f"Operation failed: {error_msg}")
    
    resp = op.get("response", {})
    
    # Check for RAI filtering
    rai_filtered = resp.get("raiMediaFilteredCount", 0)
    if rai_filtered > 0:
        reasons = resp.get("raiMediaFilteredReasons", [])
        print(f"   [WARN] {rai_filtered} video(s) filtered due to safety policies: {reasons}")
        return None
    
    # Get videos array
    videos = resp.get("videos", [])
    if not videos:
        print("   [WARN] No videos in response")
        return None
    
    # Get first video
    video = videos[0]
    
    # Check for base64 encoded video
    if video.get("bytesBase64Encoded"):
        try:
            video_bytes = base64.b64decode(video["bytesBase64Encoded"])
            size_mb = len(video_bytes) / (1024 * 1024)
            print(f"   [INFO] Received video ({size_mb:.2f} MB)")
            return video_bytes
        except Exception as e:
            print(f"   [WARN] Failed to decode base64 video: {e}")
            return None
    
    # Check for GCS URI
    if video.get("gcsUri"):
        gcs_uri = video["gcsUri"]
        print(f"   [INFO] Video stored at: {gcs_uri}")
        print(f"   [WARN] GCS download not implemented, skipping...")
        return None
    
    print("   [WARN] Video object has no bytesBase64Encoded or gcsUri field")
    return None


def _extract_last_frame(video_path: str, frame_path: str) -> bool:
    """Extract the last frame of a video using ffmpeg."""
    try:
        Path(frame_path).parent.mkdir(parents=True, exist_ok=True)
        # Use ffmpeg to get the last frame
        cmd = [
            "ffmpeg",
            "-sseof", "-0.1",  # seek to near the end
            "-i", video_path,
            "-frames:v", "1",
            "-y",
            frame_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and Path(frame_path).exists():
            return True
        return False
    except Exception:
        return False


def generate_beat_video(
    beat: Dict[str, Any],
    character_image_path: str,
    continuity_frame_path: Optional[str],
    setting: str,
    cinematic_style: str,
    output_path: str,
    aspect_ratio: str = "9:16",
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    model: str = "veo-3.0-generate-001"
) -> bool:
    """
    Generate a video for a single beat using its character image.
    
    Args:
        beat: Beat dict with 'id', 'duration', 'character_action'
        character_image_path: Path to the character image for this beat
        setting: Story setting description
        cinematic_style: Cinematic style description
        output_path: Where to save the output video
        aspect_ratio: Video aspect ratio
        project_id: GCP project ID (defaults to DEFAULT_PROJECT_ID)
        location: GCP location (defaults to DEFAULT_LOCATION)
        model: Veo model to use
    
    Returns:
        True if successful, False otherwise
    """
    project_id = project_id or DEFAULT_PROJECT_ID
    location = location or DEFAULT_LOCATION
    
    beat_id = beat.get("id")
    duration = beat.get("duration", 6)
    action = beat.get("character_action", "")
    
    # Adjust duration to valid Veo values (4, 6, or 8 seconds for Veo 3)
    if duration < 5:
        veo_duration = 4
    elif duration <= 6:
        veo_duration = 6
    else:
        veo_duration = 8
    
    # Build comprehensive prompt
    prompt = (
        f"Animate this character performing the following action:\n"
        f"{action}\n\n"
        f"Setting: {setting}\n"
        f"Style: {cinematic_style}\n"
        f"Maintain character consistency, smooth camera movements, cinematic lighting."
    )
    
    print(f"\n{'='*70}")
    print(f"[BEAT {beat_id}] Generating video ({veo_duration}s)")
    print(f"{'='*70}")
    print(f"[INPUT] Character image: {character_image_path}")
    print(f"[INPUT] Action: {action}")
    print(f"[INPUT] Duration: {duration}s -> {veo_duration}s (Veo adjusted)")
    
    try:
        # Start the job
        print(f"[START] Initiating video generation...")
        op_data = _start_video_job(
            project_id=project_id,
            location=location,
            model=model,
            prompt=prompt,
            image_path=continuity_frame_path or character_image_path,
            duration=veo_duration,
            aspect_ratio=aspect_ratio
        )
        
        op_name = op_data.get("name", "unknown")
        print(f"[START] Operation created: {op_name.split('/')[-1]}")
        
        # Check if already done
        if op_data.get("done"):
            print("[POLL] Operation completed immediately!")
            op = op_data
        else:
            print("[POLL] Waiting for completion (typically 2-5 minutes)...")
            op = _poll_operation(project_id, location, model, op_name)
        
        # Extract video
        video_bytes = _extract_video_bytes(op)
        
        if video_bytes:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "wb") as f:
                f.write(video_bytes)
            print(f"[SUCCESS] Video saved: {output_path}")
            # Extract last frame for continuity to next beat
            next_frame = str(Path(output_path).with_suffix(".last_frame.png"))
            if _extract_last_frame(output_path, next_frame):
                print(f"   [INFO] Saved last frame for continuity: {next_frame}")
            return True
        else:
            print(f"[FAIL] No video bytes received")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to generate beat {beat_id}: {e}")
        return False


def generate_story_videos(
    story_json_path: str = "story.json",
    characters_dir: str = "characters",
    output_dir: str = "videos",
    aspect_ratio: str = "9:16",
    model: str = "veo-3.0-generate-001"
) -> List[str]:
    """
    Generate videos for all beats in the story.
    
    Args:
        story_json_path: Path to story JSON file
        characters_dir: Directory containing character images
        output_dir: Directory to save output videos
        aspect_ratio: Video aspect ratio ('9:16' or '16:9')
        model: Veo model to use
    
    Returns:
        List of successfully generated video paths
    """
    # Resolve paths
    story_path = Path(story_json_path)
    if not story_path.is_absolute():
        backend_dir = Path(__file__).resolve().parent
        root_dir = backend_dir.parent
        candidates = [backend_dir / story_json_path, root_dir / story_json_path]
        story_path = next((p for p in candidates if p.exists()), candidates[0])
    
    # Load story
    print(f"\n{'='*70}")
    print(f"LOADING STORY")
    print(f"{'='*70}")
    print(f"Story file: {story_path}")
    
    with open(story_path, "r", encoding="utf-8") as f:
        story = json.load(f)
    
    setting = story.get("setting", "")
    cinematic_style = story.get("cinematic_style", "")
    beats = story.get("beats", [])
    
    print(f"Total beats: {len(beats)}")
    print(f"Setting: {setting}")
    print(f"Style: {cinematic_style}")
    
    # Force 9:16 for reels format (override cinematic_style if needed)
    print(f"Forcing aspect ratio to: {aspect_ratio} (for reels format)")
    
    # Generate videos for each beat
    successful_videos = []
    failed_beats = []
    
    last_frame_for_next: Optional[str] = None
    for beat in beats:
        beat_id = beat.get("id")
        
        # Find character image (only required for beat 1; subsequent beats can rely on last-frame continuity)
        char_image = Path(characters_dir) / f"beat_{beat_id}_character.png"
        if not char_image.exists() and beat_id == 1:
            print(f"\n[SKIP] Beat {beat_id}: Character image not found at {char_image}")
            failed_beats.append(beat_id)
            continue
        
        # Output path
        output_path = Path(output_dir) / f"beat_{beat_id}.mp4"
        
        # Generate video
        success = generate_beat_video(
            beat=beat,
            character_image_path=str(char_image) if char_image.exists() else str(characters_dir),
            continuity_frame_path=last_frame_for_next,
            setting=setting,
            cinematic_style=cinematic_style,
            output_path=str(output_path),
            aspect_ratio=aspect_ratio,
            model=model
        )
        
        if success:
            successful_videos.append(str(output_path))
            # Update continuity frame for next beat
            lf = str(Path(output_path).with_suffix(".last_frame.png"))
            last_frame_for_next = lf if Path(lf).exists() else None
        else:
            failed_beats.append(beat_id)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Successful: {len(successful_videos)}/{len(beats)} beats")
    if successful_videos:
        print(f"\nGenerated videos:")
        for video in successful_videos:
            print(f"  - {video}")
    
    if failed_beats:
        print(f"\nFailed beats: {failed_beats}")
    
    return successful_videos


def stitch_videos(video_paths: List[str], output_path: str, use_ffmpeg: bool = True) -> bool:
    """
    Stitch multiple videos together into a single video.
    
    Args:
        video_paths: List of video file paths to stitch together
        output_path: Path to save the final stitched video
        use_ffmpeg: If True, use ffmpeg for stitching; if False, try moviepy
    
    Returns:
        True if successful, False otherwise
    """
    if not video_paths:
        print("[ERROR] No videos to stitch")
        return False
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"STITCHING VIDEOS")
    print(f"{'='*70}")
    print(f"Input videos: {len(video_paths)}")
    for i, path in enumerate(video_paths, 1):
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"  {i}. {path} ({size_mb:.2f} MB)")
        else:
            print(f"  {i}. {path} [NOT FOUND]")
    print(f"Output: {output_path}")
    
    if use_ffmpeg:
        return _stitch_with_ffmpeg(video_paths, output_path)
    else:
        return _stitch_with_moviepy(video_paths, output_path)


def _stitch_with_ffmpeg(video_paths: List[str], output_path: str) -> bool:
    """Stitch videos using ffmpeg (faster and more reliable)"""
    try:
        print("\n[STITCH] Using ffmpeg...")
        
        # Create a temporary file list for ffmpeg
        temp_list = Path(output_path).parent / "concat_list.txt"
        with open(temp_list, "w", encoding="utf-8") as f:
            for video_path in video_paths:
                # Convert to absolute path and use forward slashes for ffmpeg
                abs_path = Path(video_path).resolve()
                f.write(f"file '{abs_path}'\n")
        
        # Use ffmpeg concat demuxer (fastest method for same codec)
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(temp_list),
            "-c", "copy",  # Copy codec without re-encoding (fastest)
            "-y",  # Overwrite output file
            str(output_path)
        ]
        
        print(f"[STITCH] Running: {' '.join(cmd[:4])}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up temp file
        try:
            temp_list.unlink()
        except:
            pass
        
        if result.returncode == 0:
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            print(f"[SUCCESS] Stitched video saved: {output_path} ({size_mb:.2f} MB)")
            return True
        else:
            print(f"[ERROR] ffmpeg failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("[WARN] ffmpeg not found, trying moviepy...")
        return _stitch_with_moviepy(video_paths, output_path)
    except Exception as e:
        print(f"[ERROR] ffmpeg stitching failed: {e}")
        return False


def _stitch_with_moviepy(video_paths: List[str], output_path: str) -> bool:
    """Stitch videos using moviepy (fallback if ffmpeg not available)"""
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        
        print("\n[STITCH] Using moviepy...")
        
        # Load all video clips
        clips = []
        for path in video_paths:
            print(f"[STITCH] Loading {path}...")
            clip = VideoFileClip(path)
            clips.append(clip)
        
        # Concatenate
        print("[STITCH] Concatenating clips...")
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Write output
        print(f"[STITCH] Writing output to {output_path}...")
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            fps=30
        )
        
        # Clean up
        for clip in clips:
            clip.close()
        final_clip.close()
        
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] Stitched video saved: {output_path} ({size_mb:.2f} MB)")
        return True
        
    except ImportError:
        print("[ERROR] moviepy not installed. Install with: pip install moviepy")
        return False
    except Exception as e:
        print(f"[ERROR] moviepy stitching failed: {e}")
        return False


if __name__ == "__main__":
    # Get configuration from environment or use defaults
    story_path = os.getenv("STORY_PATH", "story.json")
    characters_dir = os.getenv("CHARACTERS_DIR", "characters")
    output_dir = os.getenv("OUTPUT_DIR", "videos")
    aspect_ratio = os.getenv("ASPECT_RATIO", "9:16")
    model = os.getenv("VIDEO_MODEL", "veo-3.0-generate-001")
    
    print(f"""
{'='*70}
VIDEO GENERATION PIPELINE
{'='*70}
Configuration:
  Story: {story_path}
  Characters: {characters_dir}
  Output: {output_dir}
  Aspect Ratio: {aspect_ratio}
  Model: {model}
{'='*70}
""")
    
    try:
        videos = generate_story_videos(
            story_json_path=story_path,
            characters_dir=characters_dir,
            output_dir=output_dir,
            aspect_ratio=aspect_ratio,
            model=model
        )
        
        if videos:
            print(f"\n[SUCCESS] Generated {len(videos)} video(s)")
            sys.exit(0)
        else:
            print(f"\n[FAIL] No videos were generated")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

