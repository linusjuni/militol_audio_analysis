import os, json, re
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv(), override=False)
ASR_JSON = Path("data/processed/asr/test_speech/asr_segments.json")
BG_JSON  = Path("data/processed/bg/test_speech/events.json")  # or timeline.json
REPORTS_DIR = Path("data/processed/reports")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Create a .env in your project root with OPENAI_API_KEY=..."
    )

client = OpenAI(api_key=api_key)

def load_json(p):
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)

def concat_transcript(asr):
    segs = asr.get("segments", [])
    if not segs:
        return ""
    lines = []
    for s in segs:
        spk = s.get("speaker", "S?")
        lines.append(f"[{s.get('start_s',0):.1f}-{s.get('end_s',0):.1f}] {spk}: {s.get('text','')}")
    return "\n".join(lines)

def normalize_bg_events(bg_sparse: dict) -> dict:
    """Accepts bg_sparse={"events":[...], "meta":{...}} and normalizes:
       - label lowercased
       - t or start_s/end_s → start_s/end_s
       - p or prob → p (float or None)
       - sorted by time
    """
    evs = bg_sparse.get("events", [])
    norm = []
    for e in evs:
        label = str(e.get("label","")).lower()
        if "start_s" in e or "end_s" in e:
            start = float(e.get("start_s", e.get("end_s", 0.0)))
            end   = float(e.get("end_s", e.get("start_s", start)))
        else:
            t = float(e.get("t", 0.0))
            start = end = t
        p = e.get("p", e.get("prob", None))
        p = None if p is None else float(p)
        norm.append({"label": label, "start_s": start, "end_s": end, "p": p})
    norm.sort(key=lambda x: (x["start_s"], x["end_s"]))
    return {"events": norm, "meta": bg_sparse.get("meta", {})}

def build_intel_json(clip, asr, bg):
    segs = asr.get("segments", [])
    return {
        "clip_id": clip,
        "transcript_segments": segs,
        "background": bg  
    }

def llm_report(intel, model="gpt-4o-mini"):
    system = (
        "You are an intelligence analyst. Use ONLY the JSON provided. "
        "If something is missing, say so explicitly. Include uncertainties. "
        "Ground each claim with specific timestamps from the evidence where possible."
    )
    user = (
        "You receive:\n"
        "- transcript_segments: list of {start_s, end_s, speaker, text}\n"
        "- background.events: list of {label, start_s/end_s or t, probability}\n\n"
        "Write a concise Markdown report with these sections and headings:\n\n"
        "## Executive Summary\n"
        "- 3-6 bullets with the most important findings.\n\n"
        "## Scene Inference\n"
        "- Infer likely environment(s) from background events; cite strongest evidence with timestamps and probabilities; state confidence.\n\n"
        "## Civilians\n"
        "- Assess likelihood of civilians/children being present using events (e.g., crowd/child-like sounds). Cite timestamps and probabilities; state confidence.\n\n"
        "## Contradictions\n"
        "- Compare any location/setting claims in the transcript (e.g., 'coast', 'city', 'transport') against background events. "
        "Flag contradictions with evidence and timestamps. If none, say 'None observed'.\n\n"
        "## Evidence Timeline\n"
        "- Chronological list of the most relevant background events with time ranges and probabilities; include short notes if they relate to what’s being said.\n\n"
        "JSON:\n" + json.dumps(intel, ensure_ascii=False)
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
    )
    return resp.choices[0].message.content or ""


def generate_report(asr_json_path, bg_json_path, out_dir, model = "gpt-4o-mini"):
    asr = load_json(asr_json_path)
    bg  = load_json(bg_json_path if bg_json_path.exists() else bg_json_path.with_name("timeline.json"))
    clip_name = asr_json_path.parent.name
    intel = build_intel_json(clip_name, asr, bg)
    report = llm_report(intel, model=model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_name}.md"
    out_path.write_text(report)
    return out_path, report