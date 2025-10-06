from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class Beat(BaseModel):
    id: int = Field(..., description="Sequential beat id starting at 1")
    character_action: str = Field(..., description="What the character does/feels in this beat")
    duration: int = Field(..., description="Beat duration in seconds")

class StoryOutput(BaseModel):
    theme: str
    total_duration: int
    main_character: str
    setting: str
    cinematic_style: str
    beats: List[Beat]

def generate_story_with_langchain(theme: str, total_duration:int) -> Dict[str, Any]:
    """
    Generates a structured storyboard JSON for a silent 30–60s reel using OpenAI GPT-4o-mini via LangChain.
    If theme is "auto"/empty, it first asks the LLM to propose a concise theme.
    """

    # Auto-pick theme if requested
    if not theme or theme.lower() in ("auto", "", "default"):
        try:
            theme = suggest_theme_with_langchain()
        except Exception:
            theme = "rediscovering joy after heartbreak"

    # Define LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Prompt template
    template = """
You are a cinematic storyteller and creative director. Generate a highly visual, continuous storyboard for a 30–60 second **silent reel** — no dialogue, no voice, no text overlays.
Theme: "{theme}"
Total Duration: {total_duration} seconds (the final video must last exactly this long).

Your task:
1️ Define ONE consistent MAIN CHARACTER for the entire reel. Describe physical appearance, outfit, and vibe in 2 lines.
2️ Define ONE consistent SETTING/ENVIRONMENT for the reel. Describe it vividly — include atmosphere, time of day, lighting, and tone.
3️ Define ONE consistent CINEMATIC STYLE for the reel. Mention visual tone, camera work, lighting style, and color palette (e.g., “warm tones, handheld, shallow depth of field, cinematic backlight”).
4️ Divide the reel into narrative BEATS (moments). 
   - You must decide how many beats (usually 5–8).
   - Each beat lasts between 3–6 seconds.
   - Total duration of all beats must equal exactly {total_duration}.
   - Each beat describes ONLY the **character’s actions, emotions, and camera framing**, keeping setting and style consistent.

When describing each beat:
- Use clear visual verbs (e.g., “looks up,” “turns slowly,” “walks through the mist,” “clenches fist,” “smiles faintly,” “leans against railing”).
- Include emotional cues, lighting direction, and approximate shot framing (close-up, medium, wide, over-the-shoulder, aerial, etc.).
- Each beat should feel like a single cinematic frame that could be used for **image generation**.

🎥 Example Output:
{{
  "theme": "rediscovering joy after heartbreak",
  "total_duration": 40,
  "main_character": "A young woman in her mid-20s, long dark hair tied back, wearing a soft blue dress and white sneakers, gentle expression but tired eyes.",
  "setting": "A quiet neon-lit city street at night after rain — glowing reflections on wet pavement, occasional passing cars, warm shop lights, cool mist in the air.",
  "cinematic_style": "9:16 aspect ratio, handheld camera, shallow depth of field, warm-cool color contrast, cinematic lens flare, soft ambient rain sounds.",
  "beats": [
    {{
      "id": 1,
      "character_action": "CLOSE-UP — The woman sits on the wet sidewalk curb, head lowered, droplets of rain running down her face, city lights reflecting in her teary eyes.",
      "duration": 6
    }},
    {{
      "id": 2,
      "character_action": "WIDE SHOT — She slowly stands and begins walking under flickering streetlights, her reflection visible in the puddles, camera tracking from behind.",
      "duration": 7
    }},
    {{
      "id": 3,
      "character_action": "MEDIUM SHOT — She pauses outside a small café window, the warm golden light spilling onto her face as she watches people laughing inside.",
      "duration": 7
    }},
    {{
      "id": 4,
      "character_action": "CLOSE-UP — Her hand trembles slightly as she grips the door handle, takes a breath, and steps inside, soft light illuminating her silhouette.",
      "duration": 10
    }},
    {{
      "id": 5,
      "character_action": "WIDE SHOT — The woman exits the café with a friend beside her, faint smile returning; the camera slowly cranes upward, revealing the glowing city skyline.",
      "duration": 10
    }}
  ]
}}

⚠️ Output ONLY valid JSON strictly in this format.
"""

    prompt = PromptTemplate.from_template(template)

    # JSON parser to validate output
    parser = JsonOutputParser(pydantic_object=StoryOutput)

    # Build chain: prompt → llm → parser
    chain = prompt | llm | parser

    # Execute chain with inputs
    result = chain.invoke({
        "theme": theme,
        "total_duration": total_duration
    })
    # result is a dict validated by StoryOutput
    return result.model_dump() if isinstance(result, BaseModel) else result


def suggest_theme_with_langchain() -> str:
    """Ask the LLM to propose a single concise cinematic theme.

    Returns a short text like: "rediscovering joy after burnout".
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
    prompt = (
        "Suggest one creative cinematic theme for a 30–60s silent vertical reel. "
        "Output ONLY the theme text. No quotes, no extra words."
    )
    msg = HumanMessage(content=prompt)
    try:
        resp = llm.invoke([msg])
        text = (resp.content or "").strip()
        # Guardrails: keep it short and single line
        return text.splitlines()[0][:120] if text else "rediscovering joy after heartbreak"
    except Exception:
        # Fallback theme if LLM call fails
        return "rediscovering joy after heartbreak"
