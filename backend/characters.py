import os
import json
import base64
from pathlib import Path
import vertexai
from google.api_core import exceptions as gax_exceptions
import time
from vertexai.preview.vision_models import ImageGenerationModel
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from glob import glob
from typing import Optional


# Lazy initialization for Vertex AI and the Imagen model
DEFAULT_PROJECT_ID = (
    os.getenv("GOOGLE_CLOUD_PROJECT")
    or os.getenv("GCP_PROJECT")
    or os.getenv("GCLOUD_PROJECT")
    or "reel-automation-using-vertex"
)
DEFAULT_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# Seed to improve character identity consistency across beats
IDENTITY_SEED = int(os.getenv("IDENTITY_SEED", "1337"))

_vertexai_initialized = False
_imagen_model = None

def _init_vertex_ai_if_needed():
    global _vertexai_initialized
    if _vertexai_initialized:
        return

    credentials = None

    # Try explicit env var first
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.exists(credentials_path):
        try:
            credentials = ServiceAccountCredentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        except Exception:
            credentials = None

    # If not set or failed, try common local paths and root json files
    if credentials is None:
        backend_dir = Path(__file__).resolve().parent
        root_dir = backend_dir.parent
        candidate_paths = [
            backend_dir / "gcp-key.json",
            root_dir / "gcp-key.json",
            root_dir / "service-account.json",
        ]
        # Add any JSON files in project root as last resort candidates
        try:
            for json_path in glob(str(root_dir / "*.json")):
                candidate_paths.append(Path(json_path))
        except Exception:
            pass

        for candidate in candidate_paths:
            try:
                if candidate and candidate.exists():
                    credentials = ServiceAccountCredentials.from_service_account_file(
                        str(candidate),
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                    # Ensure downstream libraries that rely on ADC discover the same key
                    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate)
                    break
            except Exception:
                continue

    # If credentials is None, Vertex AI will fall back to ADC if available
    # Make sure project env var is visible to libs that attempt to auto-detect
    if not os.getenv("GOOGLE_CLOUD_PROJECT") and DEFAULT_PROJECT_ID:
        os.environ["GOOGLE_CLOUD_PROJECT"] = DEFAULT_PROJECT_ID

    vertexai.init(project=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION, credentials=credentials)
    _vertexai_initialized = True

def get_imagen_model() -> ImageGenerationModel:
    global _imagen_model
    if _imagen_model is not None:
        return _imagen_model
    _init_vertex_ai_if_needed()
    _imagen_model = ImageGenerationModel.from_pretrained("imagegeneration@005")
    return _imagen_model

# -------------------------------------------------
# Helper to ensure directory exists
# -------------------------------------------------
def ensure_parent_dir(path: str):
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)

# -------------------------------------------------
# Helper to safely generate images with retries
# -------------------------------------------------
def _safe_generate_images(
    model: ImageGenerationModel,
    prompt: str,
    number_of_images: int = 1,
    aspect_ratio: str = "9:16",
    max_retries: int = 2,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
):
    attempts = 0
    last_error = None
    current_prompt = prompt
    while attempts <= max_retries:
        try:
            # Prefer passing identity controls if supported by SDK; otherwise fallback
            try:
                return model.generate_images(
                    prompt=current_prompt,
                    number_of_images=number_of_images,
                    aspect_ratio=aspect_ratio,
                    negative_prompt=negative_prompt,
                    seed=seed,
                )
            except TypeError:
                return model.generate_images(
                    prompt=current_prompt,
                    number_of_images=number_of_images,
                    aspect_ratio=aspect_ratio,
                )
        except gax_exceptions.InvalidArgument as e:
            last_error = e
            message = str(e)
            # Retry without seed if watermark blocks seeded generation
            if seed is not None and "Seed is not supported when watermark is enabled" in message:
                try:
                    return model.generate_images(
                        prompt=current_prompt,
                        number_of_images=number_of_images,
                        aspect_ratio=aspect_ratio,
                        negative_prompt=negative_prompt,
                    )
                except Exception as e2:
                    last_error = e2
                    time.sleep(0.5)
                    attempts += 1
                    continue
            if "blocked" in message.lower() or "violat" in message.lower():
                # Sanitize prompt and retry
                if attempts == 0:
                    current_prompt = (
                        "Create a wholesome, safe-for-work, non-sensitive image. "
                        "Avoid any content that may violate safety policies.\n" + prompt
                    )
                else:
                    current_prompt = "Create a wholesome, neutral, non-sensitive portrait of the same character performing the action in a safe and policy-compliant manner."
                time.sleep(1.0)
                attempts += 1
                continue
            # Other InvalidArgument: backoff and retry without changing prompt
            time.sleep(0.5)
            attempts += 1
            continue
        except Exception as e:
            last_error = e
            time.sleep(0.5)
            attempts += 1
            continue
    if last_error:
        raise last_error
    return []

# -------------------------------------------------
# Step 1: Generate reference character
# -------------------------------------------------
def generate_reference_character(main_character: str, output_dir="characters"):
    output_path = os.path.join(output_dir, "character_reference.png")

    prompt = f"""
    Generate the MAIN CHARACTER for this reel:
    {main_character}.
    9:16 aspect ratio, cinematic high-quality portrait, full-body, facing camera.
    Realistic, consistent identity for use across multiple scenes.
    Background removed (transparent if possible).
    """

    negative_prompt = (
        "different person, changed face, different hairstyle, different hair color, different clothing, "
        "cartoonish, deformed, cropped, extra people"
    )

    print("🎨 Generating reference character...")
    images = _safe_generate_images(
        get_imagen_model(),
        prompt=prompt,
        number_of_images=1,
        aspect_ratio="9:16",
        negative_prompt=negative_prompt,
        seed=IDENTITY_SEED,
    )

    ensure_parent_dir(output_path)
    try:
        img = images[0]
    except Exception:
        # Fallback: retry without seed and negative constraints
        fallback = _safe_generate_images(
            get_imagen_model(),
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="9:16",
            negative_prompt=None,
            seed=None,
        )
        try:
            img = fallback[0]
        except Exception:
            raise RuntimeError("No image returned for reference character generation")
    img.save(location=output_path)
    print(f"✅ Image saved: {output_path}")
    return output_path

# -------------------------------------------------
# Step 2: Generate per-beat character variation
# -------------------------------------------------
def generate_character_variation(main_character: str, action: str, beat_id: int, output_dir="characters"):
    output_path = os.path.join(output_dir, f"beat_{beat_id}_character.png")

    prompt = f"""
    Generate the EXACT SAME main character as in the reference image performing the described action.
    Character identity must remain identical: same face, facial structure, eye color, hairstyle, hair color,
    clothing/outfit, and body proportions. Do NOT change identity.

    Action: {action}.
    9:16 aspect ratio, cinematic lighting, realistic art style.
    Transparent background if possible.
    """

    negative_prompt = (
        "different person, changed identity, different face, different hairstyle, different hair color, "
        "different clothing/outfit, different age, different ethnicity, extra people, cropped face, profile-only"
    )

    print(f"🎬 Generating variation for beat {beat_id}: {action}")
    images = _safe_generate_images(
        get_imagen_model(),
        prompt=prompt,
        number_of_images=1,
        aspect_ratio="9:16",
        negative_prompt=negative_prompt,
        seed=IDENTITY_SEED + beat_id,
    )

    ensure_parent_dir(output_path)
    try:
        img = images[0]
    except Exception:
        # Fallback: retry without seed/negative constraints to avoid edge cases
        fallback = _safe_generate_images(
            get_imagen_model(),
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="9:16",
            negative_prompt=None,
            seed=None,
        )
        try:
            img = fallback[0]
        except Exception:
            raise RuntimeError(f"No image returned for beat {beat_id} variation")
    img.save(location=output_path)
    print(f"✅ Image saved: {output_path}")
    return output_path

# -------------------------------------------------
# Step 3: Generate all characters
# -------------------------------------------------
def generate_all_characters(story_json_path="story_output.json", output_dir="characters"):
    # Resolve path relative to project root (one level above this file)
    story_path = Path(story_json_path)
    if not story_path.is_absolute():
        backend_dir = Path(__file__).resolve().parent
        root_dir = backend_dir.parent
        candidates = [backend_dir / story_json_path, root_dir / story_json_path]
        story_path = next((p for p in candidates if p.exists()), candidates[0])
    with open(story_path, "r", encoding="utf-8") as f:
        story = json.load(f)

    if not isinstance(story, dict):
        raise ValueError("Story JSON must be an object with keys 'main_character' and 'beats'.")

    if "main_character" not in story or not story["main_character"]:
        raise KeyError("'main_character' is missing or empty in story JSON")
    main_character = story["main_character"]

    # Step 1: Reference character
    generate_reference_character(main_character, output_dir)

    # Step 2: Variations for each beat
    beats = story.get("beats", [])
    if not isinstance(beats, list):
        raise ValueError("'beats' must be a list in story JSON")
    for beat in beats:
        action = beat.get("character_action", "Performs an action")
        beat_id = beat.get("id", 0)
        generate_character_variation(main_character, action, beat_id, output_dir)

    print("✅ All character images generated successfully.")


def generate_minimal_characters(story_json_path="story.json", output_dir="characters"):
    """Generate only reference + beat 1 character (sufficient for continuity-chained videos)."""
    # Resolve path relative to project root (one level above this file)
    story_path = Path(story_json_path)
    if not story_path.is_absolute():
        backend_dir = Path(__file__).resolve().parent
        root_dir = backend_dir.parent
        candidates = [backend_dir / story_json_path, root_dir / story_json_path]
        story_path = next((p for p in candidates if p.exists()), candidates[0])
    with open(story_path, "r", encoding="utf-8") as f:
        story = json.load(f)

    if not isinstance(story, dict):
        raise ValueError("Story JSON must be an object with keys 'main_character' and 'beats'.")

    if "main_character" not in story or not story["main_character"]:
        raise KeyError("'main_character' is missing or empty in story JSON")
    main_character = story["main_character"]

    # Reference character
    generate_reference_character(main_character, output_dir)

    # Beat 1 only
    beats = story.get("beats", [])
    if not isinstance(beats, list) or not beats:
        raise ValueError("'beats' must be a non-empty list in story JSON")
    first = beats[0]
    action = first.get("character_action", "Performs an action")
    beat_id = first.get("id", 1)
    generate_character_variation(main_character, action, beat_id, output_dir)
    print("✅ Minimal character set generated: reference + beat_1 only.")


if __name__ == "__main__":
    generate_all_characters("story.json")
