#!/usr/bin/env python3
"""
Auto Reel Generator - Main Workflow Orchestrator

This script orchestrates the complete workflow:
1. Generate story from theme
2. Generate character images for each beat
3. Generate videos for each beat using character images
4. Stitch videos together into final reel
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.story_gen import generate_story_with_langchain, suggest_theme_with_langchain
from backend.characters import generate_all_characters, generate_minimal_characters
from backend.video import generate_story_videos, stitch_videos


def print_banner(title: str):
    """Print a formatted banner"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_step(step_num: int, step_name: str):
    """Print a step header"""
    print(f"\n{'*'*70}")
    print(f"STEP {step_num}: {step_name}")
    print(f"{'*'*70}\n")


def generate_reel(
    theme: str,
    duration: int = 45,
    story_output: str = "story.json",
    characters_dir: str = "characters",
    videos_dir: str = "videos",
    final_output: str = "final_reel.mp4",
    aspect_ratio: str = "9:16",
    video_model: str = "veo-3.0-generate-001",
    skip_story: bool = False,
    skip_characters: bool = False,
    skip_videos: bool = False
) -> Optional[str]:
    """
    Generate a complete reel from a theme.
    
    Args:
        theme: Theme for the story (e.g., "overcoming heartbreak")
        duration: Total duration in seconds
        story_output: Path to save story JSON
        characters_dir: Directory to save character images
        videos_dir: Directory to save individual beat videos
        final_output: Path to save final stitched reel
        aspect_ratio: Video aspect ratio (9:16 for reels, 16:9 for landscape)
        video_model: Veo model to use
        skip_story: Skip story generation (use existing story.json)
        skip_characters: Skip character generation (use existing images)
        skip_videos: Skip video generation (use existing videos)
    
    Returns:
        Path to final reel if successful, None otherwise
    """
    start_time = time.time()
    
    print_banner("AUTO REEL GENERATOR")
    print(f"Theme: {theme}")
    print(f"Duration: {duration} seconds")
    print(f"Aspect Ratio: {aspect_ratio}")
    print(f"Video Model: {video_model}")
    print(f"Output: {final_output}")
    
    # Resolve paths
    story_path = Path(story_output)
    if not story_path.is_absolute():
        backend_dir = Path(__file__).resolve().parent
        root_dir = backend_dir.parent
        story_path = root_dir / story_output
    
    try:
        # Step 1: Generate Story
        if not skip_story:
            print_step(1, "GENERATING STORY")
            print(f"Theme: {theme}")
            print(f"Target duration: {duration}s")
            
            story = generate_story_with_langchain(theme, duration)
            
            with open(story_path, "w", encoding="utf-8") as f:
                json.dump(story, f, indent=2)
            
            print(f"\n[SUCCESS] Story generated with {len(story.get('beats', []))} beats")
            print(f"[SUCCESS] Story saved to: {story_path}")
        else:
            print_step(1, "LOADING EXISTING STORY")
            if not story_path.exists():
                print(f"[ERROR] Story file not found: {story_path}")
                return None
            with open(story_path, "r", encoding="utf-8") as f:
                story = json.load(f)
            print(f"[SUCCESS] Loaded story with {len(story.get('beats', []))} beats")
        
        # Step 2: Generate Characters (minimal)
        if not skip_characters:
            print_step(2, "GENERATING CHARACTER IMAGES (minimal)")
            print(f"Output directory: {characters_dir}")
            
            generate_minimal_characters(
                story_json_path=str(story_path),
                output_dir=characters_dir
            )
            
            print(f"\n[SUCCESS] Character images generated (reference + beat_1)")
        else:
            print_step(2, "USING EXISTING CHARACTER IMAGES")
            char_dir = Path(characters_dir)
            char_count = len(list(char_dir.glob("beat_*_character.png")))
            print(f"[SUCCESS] Found {char_count} character images in {characters_dir}")
        
        # Step 3: Generate Videos
        if not skip_videos:
            print_step(3, "GENERATING VIDEOS")
            print(f"Input: {characters_dir}")
            print(f"Output: {videos_dir}")
            print(f"Aspect ratio: {aspect_ratio}")
            
            video_paths = generate_story_videos(
                story_json_path=str(story_path),
                characters_dir=characters_dir,
                output_dir=videos_dir,
                aspect_ratio=aspect_ratio,
                model=video_model
            )
            
            if not video_paths:
                print(f"[ERROR] No videos were generated")
                return None
            
            print(f"\n[SUCCESS] Generated {len(video_paths)} video(s)")
        else:
            print_step(3, "USING EXISTING VIDEOS")
            video_dir = Path(videos_dir)
            video_paths = sorted([str(p) for p in video_dir.glob("beat_*.mp4")])
            print(f"[SUCCESS] Found {len(video_paths)} video(s) in {videos_dir}")
        
        # Step 4: Stitch Videos
        print_step(4, "STITCHING FINAL REEL")
        print(f"Combining {len(video_paths)} videos")
        print(f"Output: {final_output}")
        
        success = stitch_videos(video_paths, final_output)
        
        if not success:
            print(f"[ERROR] Failed to stitch videos")
            return None
        
        # Final summary
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        final_size = Path(final_output).stat().st_size / (1024 * 1024)
        
        print_banner("COMPLETE!")
        print(f"[SUCCESS] Final reel generated: {final_output}")
        print(f"[SUCCESS] File size: {final_size:.2f} MB")
        print(f"[SUCCESS] Total time: {minutes}m {seconds}s")
        print(f"\n{'='*70}\n")
        
        return final_output
        
    except KeyboardInterrupt:
        print("\n\n[CANCELED] Process interrupted by user")
        return None
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point - runs full workflow by default"""
    # Get configuration from environment or use defaults
    theme = os.getenv("THEME", "auto")
    duration = int(os.getenv("DURATION", "45"))
    story_output = os.getenv("STORY_OUTPUT", "story.json")
    characters_dir = os.getenv("CHARACTERS_DIR", "characters")
    videos_dir = os.getenv("VIDEOS_DIR", "videos")
    final_output = os.getenv("FINAL_OUTPUT", "final_reel.mp4")
    aspect_ratio = os.getenv("ASPECT_RATIO", "9:16")
    video_model = os.getenv("VIDEO_MODEL", "veo-3.0-generate-001")
    
    # Check for skip flags (optional - for advanced users)
    skip_story = os.getenv("SKIP_STORY", "false").lower() in ("true", "1", "yes")
    skip_characters = os.getenv("SKIP_CHARACTERS", "false").lower() in ("true", "1", "yes")
    skip_videos = os.getenv("SKIP_VIDEOS", "false").lower() in ("true", "1", "yes")
    
    # Auto theme if requested or missing
    if not theme or theme.lower() in ("auto", "", "default"):
        try:
            theme = suggest_theme_with_langchain()
        except Exception:
            theme = "overcoming heartbreak and rediscovering purpose"

    print(f"""
{'='*70}
AUTO REEL GENERATOR
{'='*70}

Starting full workflow to generate your reel...

Theme: {theme}
Duration: {duration}s
Output: {final_output}
Format: {aspect_ratio} (portrait for reels)

This will take approximately 7-10 minutes.
Please wait while we generate your reel...
{'='*70}
""")
    
    result = generate_reel(
        theme=theme,
        duration=duration,
        story_output=story_output,
        characters_dir=characters_dir,
        videos_dir=videos_dir,
        final_output=final_output,
        aspect_ratio=aspect_ratio,
        video_model=video_model,
        skip_story=skip_story,
        skip_characters=skip_characters,
        skip_videos=skip_videos
    )
    
    if result:
        print(f"""
{'='*70}
SUCCESS! Your reel is ready!
{'='*70}

Final Reel: {result}

You can now:
- Open {result} to view your reel
- Upload it to Instagram Reels, TikTok, or YouTube Shorts
- Share it on social media!

{'='*70}
""")
        sys.exit(0)
    else:
        print(f"""
{'='*70}
FAILED
{'='*70}
Failed to generate reel. Check the error messages above.
{'='*70}
""")
        sys.exit(1)


if __name__ == "__main__":
    main()
