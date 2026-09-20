"""ComfyUI bridge to a Spark Studio or direct OpenAI-compatible model server."""

import base64
import io
import json
import math
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
from PIL import Image


def chat_url(base_url):
    base = str(base_url or "").strip().rstrip("/")
    parts = urlsplit(base)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("Spark Studio URL must be an http(s) host, such as http://spark-ip:8888/v1")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("Use a plain base URL without credentials, query, or fragment")
    if parts.path.endswith("/chat/completions"):
        return base
    if parts.path.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def models_url(base_url):
    """The /models endpoint that sits beside the chat endpoint."""
    chat = chat_url(base_url)
    suffix = "/chat/completions"
    return (chat[: -len(suffix)] if chat.endswith(suffix) else chat) + "/models"


def _unreachable(url, error):
    """Explain a connection failure in terms the person can act on.

    The most common mistake is pasting an example host from the README
    instead of the machine's real address, which surfaces as a DNS failure.
    """
    reason = str(getattr(error, "reason", None) or error)
    host = urlsplit(url).hostname or ""
    lowered = reason.lower()
    if "getaddrinfo" in lowered or "not known" in lowered or "nodename" in lowered:
        hint = (f"The host name {host!r} does not exist on this network. If you copied "
                "an example address, replace it with your server's real one, such as "
                "its LAN IP or Tailscale address.")
    elif "refused" in lowered:
        hint = (f"Nothing is listening on {host} at that port. Start the server, or "
                "correct the port.")
    elif "timed out" in lowered or "timeout" in lowered:
        hint = ("The host is not answering. This is usually a firewall, or a server "
                "bound to 127.0.0.1 on a different machine.")
    else:
        hint = "Check the host, port, or tunnel."
    return RuntimeError(f"Cannot reach {url}: {reason}. {hint}")


def get_json(url, headers, timeout_seconds):
    request = Request(url, headers=headers, method="GET")
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=int(timeout_seconds)) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read(1200).decode("utf-8", errors="replace")
        raise RuntimeError(f"Endpoint returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise _unreachable(url, error) from error


def discover_model(base_url, headers, timeout_seconds):
    """Use the first served model when the model field is left blank.

    Spark Studio retargets a stale id to whatever is live, but a plain vLLM
    or SGLang server rejects an unknown id, so ask the endpoint instead.
    """
    payload = get_json(models_url(base_url), headers, timeout_seconds)
    entries = payload.get("data") if isinstance(payload, dict) else None
    for entry in entries or []:
        if isinstance(entry, dict) and str(entry.get("id") or "").strip():
            return str(entry["id"]).strip()
    raise ValueError("The endpoint reported no models; set the model field manually")


def image_parts(images):
    if images is None:
        return []
    if hasattr(images, "detach"):
        images = images.detach().cpu().numpy()
    array = np.asarray(images)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[-1] not in (1, 3, 4):
        raise ValueError("IMAGE must be a ComfyUI [batch, height, width, channels] tensor")
    if len(array) > 4:
        raise ValueError("Connect a batch of at most four images")
    parts = []
    for frame in array:
        pixels = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        if pixels.shape[-1] == 1:
            pixels = pixels[..., 0]
        image = Image.fromarray(pixels)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        # JPEG keeps multimodal requests small enough for a LAN call.
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}})
    return parts


def extract_text(response):
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("Model returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content
                         if isinstance(part, dict) and part.get("type") == "text")
    raise ValueError("Model returned no text content; check that the selected model supports chat")


def repair_text_encoding(text):
    """Repair the replacement glyph some OpenAI-compatible gateways emit for apostrophes."""
    return str(text or "").replace("\ufffd", "'")


def _source_id(value):
    if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], (str, int)):
        return str(value[0])
    return None


def song_context(prompt_graph, unique_id):
    """Find Render's duration only when this node feeds its Compose lyrics."""
    if not isinstance(prompt_graph, dict) or unique_id is None:
        return None
    nodes = {str(key): value for key, value in prompt_graph.items() if isinstance(value, dict)}
    own_id = str(unique_id)
    if own_id not in nodes:
        return None
    reachable = {own_id}
    changed = True
    while changed:
        changed = False
        for node_id, data in nodes.items():
            if node_id in reachable or data.get("class_type") in ("FL_YuE2_Plan", "FL_YuE2_Render"):
                continue
            if any(_source_id(value) in reachable for value in data.get("inputs", {}).values()):
                reachable.add(node_id)
                changed = True
    plans = {node_id: data for node_id, data in nodes.items()
             if data.get("class_type") == "FL_YuE2_Plan"
             and _source_id(data.get("inputs", {}).get("lyrics")) in reachable}
    matches = []
    for data in nodes.values():
        if data.get("class_type") != "FL_YuE2_Render":
            continue
        inputs = data.get("inputs", {})
        plan_id = _source_id(inputs.get("composition"))
        # Upstream FL-YuE2 calls this max_duration; some forks rename it.
        duration = inputs.get("max_duration")
        if duration is None:
            duration = inputs.get("target_duration")
        if plan_id in plans and isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
            style = plans[plan_id].get("inputs", {}).get("style", "")
            matches.append((int(duration), style if isinstance(style, str) else ""))
    return min(matches, key=lambda match: match[0]) if matches else None


def lyric_budget(seconds):
    """Duration-only target range; YuE2 chooses the actual tempo later."""
    seconds = int(seconds)
    if seconds <= 0:
        return None
    singing_seconds = max(12, seconds - max(20, round(seconds * 0.15)))
    max_lines = min(80, max(6, math.floor(singing_seconds / 5.0)))
    max_words = min(max_lines * 8, math.floor(singing_seconds * 1.20))
    min_lines = max(4, round(max_lines * 0.72))
    min_words = max(24, round(max_words * 0.68))
    return min_lines, max_lines, min_words, max_words


def lyrics_fit_duration(lines, words, seconds, budget):
    """Reject both rushed walls of words and skeletal, underwritten songs."""
    singing_seconds = max(12, int(seconds) - max(20, round(int(seconds) * 0.15)))
    min_lines, max_lines, min_words, max_words = budget
    # A few words over an estimate should not force Qwen to flatten a strong
    # lyric. The tolerance is deliberately small; line count stays strict.
    max_words_with_tolerance = math.ceil(max_words * 1.05)
    return (min_lines <= lines <= max_lines and min_words <= words <= max_words_with_tolerance
            and max(words / 1.20, lines * 5.0) <= singing_seconds * 1.05)


def lyric_size(text):
    lines = [line.strip() for line in text.splitlines()
             if line.strip() and not re.fullmatch(r"\[[^\]]+\]", line.strip())]
    words = re.findall(r"\b[\w]+(?:['’][\w]+)*\b", " ".join(lines), flags=re.UNICODE)
    return len(lines), len(words)


def requested_sections(prompt, system_prompt=""):
    text = str(system_prompt or "") + "\n" + str(prompt or "")
    return {re.sub(r"\s+\d+$", "", name.strip().lower())
            for name in re.findall(r"\[([^\]]+)\]", text)
            if name.strip()}


def has_requested_sections(text, required):
    if not required:
        return True
    present = {re.sub(r"\s+\d+$", "", name.strip().lower())
               for name in re.findall(r"\[([^\]]+)\]", str(text or ""))}
    return required.issubset(present)


def concrete_section_plan(requirements):
    """Translate this workflow's explicit 2x verse/chorus request into line counts."""
    text = str(requirements or "").lower()
    has_core = all(f"[{name}]" in text for name in ("verse", "chorus", "bridge", "outro"))
    two_each = ("two verses" in text and "two choruses" in text) or "2x [verse] [chorus]" in text
    if not (has_core and two_each):
        return ""
    return ("Use exactly six blocks in this order: [verse] with 5 sung lines; [chorus] with 4 sung "
            "lines; [verse] with 5 new sung lines; [chorus] repeating the same 4-line hook; [bridge] "
            "with 3 sung lines; [outro] with 2 sung lines. That is exactly 23 sung lines total. "
            "Do not add an intro, pre-chorus, final chorus, extra section, or extra lines.")


_EMPTY_LYRIC_PLACEHOLDER = re.compile(
    r"^\s*[\[【(（]?\s*(?:空\s*歌词|empty\s+lyrics?)\s*(?:[#:]?\s*\d+)?\s*[\]】)）]?\s*$",
    re.IGNORECASE,
)


def strip_empty_lyric_placeholders(text):
    """Remove YuE/Qwen empty-slot markers without filtering real Chinese lyrics."""
    return "\n".join(line for line in str(text or "").splitlines()
                     if not _EMPTY_LYRIC_PLACEHOLDER.fullmatch(line)).strip()


def section_lines(text, count):
    cleaned = strip_empty_lyric_placeholders(text)
    lines = []
    for raw in cleaned.splitlines():
        line = raw.strip()
        if not line or re.fullmatch(r"\[[^\]]+\]", line):
            continue
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line:
            lines.append(line)
    return lines[:count] if len(lines) >= count else []


def section_is_novel(lines, prior_sections):
    prior = {re.sub(r"[^\w]+", " ", line.lower()).strip()
             for section in prior_sections.values() for line in section}
    current = {re.sub(r"[^\w]+", " ", line.lower()).strip() for line in lines}
    return not (prior & current)


def post_chat(url, payload, headers, timeout_seconds):
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=int(timeout_seconds)) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read(1200).decode("utf-8", errors="replace")
        raise RuntimeError(f"Spark Studio returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise _unreachable(url, error) from error


def section_brief(label):
    return {
        "verse 1": "Open on a concrete scene and establish the emotional situation through specific details.",
        "chorus": "Distill the song into one title-worthy central hook with emotional tension and memorable phrasing.",
        "verse 2": "Deepen the story with new details and consequences; do not rephrase Verse 1.",
        "bridge": "Create a turn, realization, or emotional peak that changes how the chorus feels.",
        "outro": "End on one resonant image or truth; provide closure without summarizing the whole song.",
    }[label]


def duration_instruction(seconds, min_lines, max_lines, min_words, max_words, style="", section_plan=""):
    style_note = style.strip()[:500]
    target_lines = round((min_lines + max_lines) / 2)
    target_words = round((min_words + max_words) / 2)
    return (f"Write a complete, release-quality song for a YuE2 render capped at {seconds} seconds, "
            "including instrumental passages. Aim for "
            f"exactly {target_lines} substantial sung lyric lines and about {target_words} sung words TOTAL "
            f"(absolute limits: {min_lines}-{max_lines} lines and {min_words}-{max_words} words), "
            "counting repeated choruses. Vary natural line lengths; do not reduce the song to fragments. "
            "Build a clear emotional arc across sections and make the chorus land with a memorable hook. "
            "Use specific sensory details, fresh imagery, conversational truth, internal rhyme where natural, "
            "and meaningful progression instead of generic filler or merely restating the request. "
            "Every line must sound like something a vocalist would actually sing.\n\n"
            "NON-LYRIC PRODUCTION METADATA (use only to shape cadence, mood, and phrasing; NEVER quote, "
            f"paraphrase, list, or sing any of these production terms): {style_note or 'not supplied'}\n\n"
            "Leave room for an instrumental intro, transitions, breaths, and a real ending. Section tags do "
            "not count as sung lines. " + (section_plan + " " if section_plan else "")
            + "Output only section tags and final singable lyrics—no commentary.")


def editor_instruction(lines, words, budget, style="", section_plan=""):
    min_lines, max_lines, min_words, max_words = budget
    target_lines = round((min_lines + max_lines) / 2)
    target_words = round((min_words + max_words) / 2)
    return ("Act as a demanding professional songwriter and revise the draft above into the final lyric. "
            "Keep the user's story, names, intimate details, language, and requested section structure, but "
            "replace bland summaries, clipped fragments, clichés, forced rhyme, and repeated filler with vivid, "
            "emotionally honest writing. Give the chorus one concise, memorable central hook and let each verse "
            "reveal something new. Do not introduce facts the user did not provide. "
            "Production metadata describes the sound only: never sing terms such as voice type, vocal range, "
            "genre, instruments, BPM, artist references, or production techniques, and never copy phrases from "
            f"this metadata: {style.strip()[:500] or 'not supplied'}. "
            f"The draft has {lines} sung lines and {words} words. Rewrite it to exactly {target_lines} "
            f"substantial sung lines and approximately {target_words} sung words. It must stay within "
            f"{min_lines}-{max_lines} lines and {min_words}-{max_words} words total, counting repetitions. "
            "Do not satisfy the count with tiny fragments; most lines should be complete, natural phrases. "
            + (section_plan + " " if section_plan else "")
            +
            "Return only section tags and polished singable lyrics.")


class SparkStudioChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Write YuE2 song lyrics only. Use [verse] and [chorus] section tags."}),
                "base_url": ("STRING", {"default": "http://localhost:7860/api/engine/v1", "tooltip": "Spark Studio gateway, or any OpenAI-compatible base such as http://spark-host:8000/v1"}),
                "model": ("STRING", {"default": "", "tooltip": "Leave blank to use the first model the endpoint serves"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE",),
                "timeout_seconds": ("INT", {"default": 180, "min": 5, "max": 3600}),
                "strip_thinking": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"prompt_graph": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate"
    CATEGORY = "text/Spark Studio"

    def generate(self, prompt, base_url, model, max_tokens, temperature, top_p,
                 system_prompt="", image=None, timeout_seconds=180, strip_thinking=True,
                 prompt_graph=None, unique_id=None):
        url = chat_url(base_url)
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("SPARK_STUDIO_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        model = str(model or "").strip()
        if not model:
            model = discover_model(base_url, headers, timeout_seconds)
            print(f"[SparkStudio] Model field blank; using {model!r}")
        context = song_context(prompt_graph, unique_id)
        budget = lyric_budget(context[0]) if context else None
        required_sections = requested_sections(prompt, system_prompt) if budget else set()
        requirements = "\n".join(part for part in (str(system_prompt).strip(), str(prompt).strip()) if part)
        section_plan = concrete_section_plan(requirements) if budget else ""
        content = str(prompt)
        if budget:
            content += "\n\n" + duration_instruction(
                context[0], *budget, style=context[1], section_plan=section_plan)
        parts = image_parts(image)
        image_note = ("IMAGE IS A PRIMARY LYRIC SOURCE: study the attached image before writing. Use only "
                      "details that are visibly supported (setting, light, colors, objects, expressions, "
                      "and mood); turn them into lyrical imagery rather than an image caption. Do not invent "
                      "identities, events, or biographical facts. If the image is unrelated or ambiguous, "
                      "follow the user's written brief instead.\n\n" if parts else "")
        if parts:
            content = [{"type": "text", "text": image_note + content}] + parts
        messages = []
        if str(system_prompt).strip():
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": content})
        payload = {
            "model": model,
            "messages": messages,
            # Bound rambling but retain enough headroom for complete sections;
            # line/word validation below enforces the actual duration budget.
            "max_tokens": (min(int(max_tokens), max(384, round(budget[3] * 2.7) + 128))
                           if budget else int(max_tokens)),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
        }
        if section_plan:
            written = {}
            plan = [("verse 1", 5), ("chorus", 4), ("verse 2", 5), ("bridge", 3), ("outro", 2)]
            for label, count in plan:
                prior = "\n\n".join(f"{name.upper()}:\n" + "\n".join(lines)
                                     for name, lines in written.items()) or "None yet."
                section_payload = dict(payload)
                section_payload["max_tokens"] = min(int(max_tokens), max(128, count * 38))
                section_payload["messages"] = [
                    {"role": "system", "content":
                     "You are an exceptional professional songwriter. Write emotionally precise, vivid, "
                     "original lyrics—not summaries, slogans, clipped fragments, or production notes. "
                     "Output only the requested lyric lines, with no tag, title, numbering, or commentary."},
                    {"role": "user", "content":
                     image_note + f"Write exactly {count} substantial lines for {label}. {section_brief(label)} "
                     "Most lines should be natural phrases of roughly 5-10 words, with varied cadence. "
                     "Use concrete sensory detail, fresh imagery, conversational truth, and rhyme only when "
                     "it sounds effortless. Preserve the user's facts; invent no new biographical facts.\n\n"
                     "PRODUCTION METADATA—sound guidance only; never quote or sing these words:\n"
                     f"{context[1].strip()[:500] or 'not supplied'}\n\n"
                     f"USER'S SONG BRIEF:\n{requirements}\n\n"
                     f"SECTIONS ALREADY WRITTEN—maintain continuity but do not repeat their ideas:\n{prior}"},
                ]
                if parts:
                    section_payload["messages"][-1]["content"] = [
                        {"type": "text", "text": section_payload["messages"][-1]["content"]}
                    ] + parts
                lines = []
                # Small sections are the easiest for the model to under-deliver;
                # allow one extra constrained retry before failing the workflow.
                for _ in range(3):
                    result = post_chat(url, section_payload, headers, timeout_seconds)
                    candidate = repair_text_encoding(extract_text(result))
                    if strip_thinking:
                        candidate = re.sub(r"(?is)<think>.*?</think>\s*", "", candidate).strip()
                    lines = section_lines(candidate, count)
                    if lines and section_is_novel(lines, written):
                        break
                    lines = []
                    retry = (f"\n\nYour last answer did not contain {count} usable lyric lines. "
                             f"Return exactly {count} complete lyric lines now. Do not copy any line from a prior section.")
                    if parts:
                        section_payload["messages"][-1]["content"][0]["text"] += retry
                    else:
                        section_payload["messages"][-1]["content"] += retry
                    section_payload["temperature"] = min(1.0, float(temperature) + 0.2)
                if not lines:
                    raise ValueError(f"Spark Studio could not write {count} usable lines for {label}")
                written[label] = lines
            text = ("[verse]\n" + "\n".join(written["verse 1"])
                    + "\n\n[chorus]\n" + "\n".join(written["chorus"])
                    + "\n\n[verse]\n" + "\n".join(written["verse 2"])
                    + "\n\n[chorus]\n" + "\n".join(written["chorus"])
                    + "\n\n[bridge]\n" + "\n".join(written["bridge"])
                    + "\n\n[outro]\n" + "\n".join(written["outro"]))
            lines, words = lyric_size(text)
            print(f"[SparkStudio] Section-written lyrics for {context[0]}s: {lines} lines, {words} words")
            return (text,)
        for attempt in range(3 if budget else 1):
            result = post_chat(url, payload, headers, timeout_seconds)
            text = repair_text_encoding(extract_text(result))
            if strip_thinking:
                text = re.sub(r"(?is)<think>.*?</think>\s*", "", text).strip()
            text = strip_empty_lyric_placeholders(text)
            if not text:
                raise ValueError("Model response contained no visible text")
            if not budget:
                return (text,)
            lines, words = lyric_size(text)
            complete = has_requested_sections(text, required_sections)
            if attempt > 0 and complete and lyrics_fit_duration(lines, words, context[0], budget):
                print(f"[SparkStudio] Lyrics fit {context[0]}s budget: {lines} lines, {words} words")
                return (text,)
            reason = ("editorial polish" if attempt == 0 else
                      "missing requested sections" if not complete else "duration/quality range")
            print(f"[SparkStudio] Revising lyrics for {reason} ({lines} lines, {words} words; "
                  f"target {budget[0]}-{budget[1]} lines, {budget[2]}-{budget[3]} words; "
                  f"attempt {attempt + 1}/3)")
            # Start a clean editorial request. Some OpenAI-compatible engines
            # strongly anchor on their prior assistant turn and echo the draft
            # even when asked to shorten it in the same conversation.
            editor_content = (editor_instruction(lines, words, budget, style=context[1], section_plan=section_plan)
                              + "\n\nORIGINAL USER REQUIREMENTS:\n" + requirements
                              + "\n\nDRAFT TO REWRITE:\n" + text)
            payload["messages"] = [
                {"role": "system", "content":
                 "You are the final lyric editor. Follow the latest numerical and editorial requirements "
                 "exactly. Return only the finished section-tagged lyrics."},
                {"role": "user", "content": image_note + editor_content},
            ]
            if parts:
                payload["messages"][-1]["content"] = [
                    {"type": "text", "text": payload["messages"][-1]["content"]}
                ] + parts
            payload["temperature"] = min(1.0, max(0.35, float(temperature) + 0.1 * attempt))
        raise ValueError(f"Spark Studio could not fit lyrics into {context[0]}s "
                         f"({budget[0]}-{budget[1]} lines/{budget[2]}-{budget[3]} words) "
                         "after 3 tries. Adjust the song idea or allow a longer render.")
