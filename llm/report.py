import os, json, re
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv(), override=False)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Create a .env in your project root with OPENAI_API_KEY=..."
    )

client = OpenAI(api_key=api_key)

def project_root(start):
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "data").exists():
            return p
    return Path.cwd().resolve()

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

def llm_report(intel, model = "gpt-4o-mini"):
    system = (
        "You are an intelligence analyst. ONLY use the JSON provided. "
        "If something is missing, say so explicitly. Include uncertainties."
    )
    user = (
        "Write a concise report with these sections:\n"
        "1) Executive Summary (3-6 bullets)\n"
        "2) Background & Civilians (what was heard; confidence; timestamps)\n"
        "3) Contradictions (claims vs audio cues; if none, say none)\n"
        "4) Speaker Dynamics (talk-time shares; any leadership cues)\n"
        "5) Evidence Timeline (key background events with times)\n\n"
        "JSON:\n" + json.dumps(intel, ensure_ascii=False)
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
    )
    return resp.choices[0].message.content or ""
