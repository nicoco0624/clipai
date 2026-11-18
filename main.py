"""
ClipAI — Transforme n'importe quelle vidéo YouTube en clips viraux
Backend : FastAPI
IA      : Claude Opus 5.5 (analyse) + Claude Sonnet 4.6 (exécution)
Video   : yt-dlp + ffmpeg + faster-whisper
"""

import os, json, uuid, asyncio, shutil, re, math, urllib.request
from pathlib import Path
from typing import Optional

# Instances Invidious publiques (proxy YouTube sans détection bot)
INVIDIOUS_INSTANCES = [
    "yewtu.be",
    "inv.nadeko.net",
    "invidious.nerdvpn.de",
    "iv.melmac.space",
    "invidious.privacydev.net",
]

def _extract_video_id(url: str) -> str:
    m = re.search(r'(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})', url)
    if not m:
        raise ValueError(f"URL YouTube invalide : {url}")
    return m.group(1)

import anthropic
import yt_dlp
from faster_whisper import WhisperModel
import ffmpeg as ffmpeg_python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CLIPS_DIR   = BASE_DIR / "clips"
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR  = BASE_DIR / "static"

CLIPS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
# Cookies YouTube encodés en base64 (variable YOUTUBE_COOKIES_B64 sur Render)
_COOKIES_B64       = os.getenv("YOUTUBE_COOKIES_B64", "")
COOKIES_PATH       = Path("/tmp/yt_cookies.txt")

if _COOKIES_B64:
    import base64
    COOKIES_PATH.write_bytes(base64.b64decode(_COOKIES_B64))
    print(f"✓  Cookies YouTube chargés ({COOKIES_PATH.stat().st_size} octets)")

MODEL_OPUS   = "claude-opus-5-5"       # analyse profonde
MODEL_SONNET = "claude-sonnet-4-6"     # exécution rapide

WHISPER_MODEL_SIZE = "base"            # base = bon équilibre vitesse/qualité sur CPU
MAX_CLIPS = 5

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="ClipAI")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static",    StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/clips",     StaticFiles(directory=CLIPS_DIR),  name="clips")

# Cache Whisper model (lourd à charger)
_whisper: Optional[WhisperModel] = None

def get_whisper() -> WhisperModel:
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu",
                                compute_type="int8")
    return _whisper

# ── State des jobs ────────────────────────────────────────────────────────
jobs: dict[str, dict] = {}

# ── Modèles API ───────────────────────────────────────────────────────────
class StartRequest(BaseModel):
    url: str
    language: str = "fr"

# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/api/start")
async def start_job(req: StartRequest, bg: BackgroundTasks):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(400, "ANTHROPIC_API_KEY non configurée. "
                            "Lance : export ANTHROPIC_API_KEY=sk-ant-...")
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "pending", "progress": 0, "step": "Démarrage…",
                    "clips": [], "error": None, "url": req.url}
    bg.add_task(process_video, job_id, req.url, req.language)
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job introuvable")
    return jobs[job_id]

@app.get("/api/download/{job_id}/{clip_id}")
def download_clip(job_id: str, clip_id: str):
    path = CLIPS_DIR / job_id / f"{clip_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "Clip introuvable")
    return FileResponse(path, media_type="video/mp4",
                        filename=f"clip_{clip_id}.mp4")

# ── Pipeline principal ────────────────────────────────────────────────────

def update(job_id: str, step: str, progress: int):
    jobs[job_id]["step"]     = step
    jobs[job_id]["progress"] = progress
    print(f"[{job_id}] {progress}% — {step}")

async def process_video(job_id: str, url: str, language: str):
    job_dir = CLIPS_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    try:
        jobs[job_id]["status"] = "running"

        # 1. Téléchargement
        update(job_id, "Téléchargement de la vidéo…", 5)
        video_path = await asyncio.to_thread(download_video, url, job_dir)

        # 2. Transcription
        update(job_id, "Transcription audio (Whisper)…", 20)
        transcript, segments = await asyncio.to_thread(transcribe, video_path)

        # 3. Analyse avec Claude Opus
        update(job_id, "Analyse IA (Claude Opus) — identification des meilleurs moments…", 40)
        clip_specs = await asyncio.to_thread(analyze_with_opus, transcript, segments, url)

        # 4. Génération des métadonnées avec Claude Sonnet
        update(job_id, "Génération des titres et sous-titres (Claude Sonnet)…", 60)
        clip_specs = await asyncio.to_thread(enrich_with_sonnet, clip_specs, language)

        # 5. Découpe + sous-titres
        clips_out = []
        total = len(clip_specs)
        for i, spec in enumerate(clip_specs):
            pct = 70 + int(25 * i / total)
            update(job_id, f"Découpe clip {i+1}/{total} : « {spec['title'][:40]} »", pct)
            clip_file = await asyncio.to_thread(
                cut_clip, video_path, spec, job_dir, i, segments
            )
            clips_out.append({
                "id":          str(i),
                "title":       spec.get("title", f"Clip {i+1}"),
                "hook":        spec.get("hook", ""),
                "hashtags":    spec.get("hashtags", []),
                "duration":    spec["end"] - spec["start"],
                "start":       spec["start"],
                "end":         spec["end"],
                "score":       spec.get("score", 8),
                "url":         f"/clips/{job_id}/{i}.mp4",
                "download":    f"/api/download/{job_id}/{i}",
                "thumbnail":   f"/clips/{job_id}/{i}_thumb.jpg",
            })

        jobs[job_id]["clips"]    = clips_out
        jobs[job_id]["status"]   = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["step"]     = f"{len(clips_out)} clip(s) prêt(s) !"

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
        print(f"[{job_id}] ERREUR : {e}")
        import traceback; traceback.print_exc()

# ── Téléchargement ────────────────────────────────────────────────────────

def _ydl_download(source_url: str, out_path: Path, cookies: bool = False) -> bool:
    """Tente un téléchargement yt-dlp. Retourne True si réussi."""
    opts = {
        "format":    "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best",
        "outtmpl":   str(out_path),
        "quiet":     True,
        "no_warnings": True,
        "merge_output_format": "mp4",
        "socket_timeout": 45,
        "postprocessors": [{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}],
    }
    if cookies and COOKIES_PATH.exists() and COOKIES_PATH.stat().st_size > 100:
        opts["cookiefile"] = str(COOKIES_PATH)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([source_url])
        return bool(list(out_path.parent.glob("source*.mp4")))
    except Exception as e:
        print(f"  yt-dlp échoué sur {source_url}: {e}")
        # nettoyer fichier partiel
        for f in out_path.parent.glob("source*"):
            f.unlink(missing_ok=True)
        return False

def download_video(url: str, out_dir: Path) -> Path:
    out_path = out_dir / "source.mp4"
    video_id = _extract_video_id(url)

    # 1. Essayer chaque instance Invidious (proxy sans bot-check)
    for host in INVIDIOUS_INSTANCES:
        inv_url = f"https://{host}/watch?v={video_id}"
        print(f"→ Essai Invidious : {host}")
        if _ydl_download(inv_url, out_path):
            candidates = list(out_dir.glob("source*.mp4"))
            if candidates:
                print(f"✓ Téléchargé via {host}")
                return candidates[0]

    # 2. Fallback direct YouTube avec cookies si disponibles
    print("→ Fallback direct YouTube")
    if _ydl_download(url, out_path, cookies=True):
        candidates = list(out_dir.glob("source*.mp4"))
        if candidates:
            return candidates[0]

    raise RuntimeError(
        "Impossible de télécharger la vidéo. "
        "YouTube bloque les IP cloud. "
        "Essaie une autre vidéo ou contacte le support."
    )

# ── Transcription ─────────────────────────────────────────────────────────

def transcribe(video_path: Path):
    model = get_whisper()
    segments_gen, _ = model.transcribe(str(video_path), beam_size=3,
                                        word_timestamps=True)
    segments = list(segments_gen)
    full_text = " ".join(s.text.strip() for s in segments)
    return full_text, segments

def segments_to_timed(segments) -> list[dict]:
    """Convertit les segments Whisper en liste {start, end, text}."""
    return [{"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in segments]

# ── Analyse Opus ──────────────────────────────────────────────────────────

OPUS_SYSTEM = """Tu es un expert en création de contenu viral pour TikTok, Instagram Reels et YouTube Shorts.
Tu analyses des transcriptions vidéo et identifies les meilleurs moments pour créer des clips courts et percutants.
Tu réponds UNIQUEMENT en JSON valide, sans aucun texte autour."""

def analyze_with_opus(transcript: str, segments, url: str) -> list[dict]:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Segments formatés avec timestamps
    timed = segments_to_timed(segments)
    duration = timed[-1]["end"] if timed else 0
    segments_text = "\n".join(
        f"[{s['start']:.1f}s-{s['end']:.1f}s] {s['text']}"
        for s in timed[:200]  # max 200 segments pour le contexte
    )

    prompt = f"""Voici la transcription d'une vidéo ({duration:.0f} secondes) :

URL : {url}

TRANSCRIPTION AVEC TIMESTAMPS :
{segments_text}

Identifie les {MAX_CLIPS} meilleurs moments pour créer des clips viraux (30-90 secondes chacun).
Critères : accroche forte, contenu autonome (compréhensible sans contexte), émotion, valeur informationnelle ou entertainment.

Réponds en JSON avec ce format exact :
{{
  "clips": [
    {{
      "start": <timestamp_debut_en_secondes>,
      "end": <timestamp_fin_en_secondes>,
      "score": <score_viralité_1_à_10>,
      "why": "<raison_en_1_phrase>"
    }}
  ]
}}"""

    response = client.messages.create(
        model=MODEL_OPUS,
        max_tokens=1500,
        system=OPUS_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    # Extraire le JSON
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        raise ValueError(f"Réponse Opus invalide : {raw[:200]}")
    data = json.loads(match.group())
    clips = data.get("clips", [])

    # Valider et corriger les timestamps
    valid = []
    for c in clips:
        s = float(c.get("start", 0))
        e = float(c.get("end", s + 60))
        if e - s < 10:
            e = s + 60
        if e - s > 120:
            e = s + 90
        valid.append({"start": s, "end": e, "score": c.get("score", 7),
                       "why": c.get("why", "")})
    return valid[:MAX_CLIPS]

# ── Enrichissement Sonnet ─────────────────────────────────────────────────

SONNET_SYSTEM = """Tu es un copywriter spécialisé contenu viral. Tu crées des titres accrocheurs,
des hooks et des sous-titres pour des clips courts. Tu réponds UNIQUEMENT en JSON valide."""

def enrich_with_sonnet(clips: list[dict], language: str) -> list[dict]:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    lang_name = "français" if language == "fr" else "anglais"

    clips_desc = json.dumps(
        [{"index": i, "start": c["start"], "end": c["end"],
          "duration": round(c["end"] - c["start"]),
          "pourquoi": c.get("why", "")}
         for i, c in enumerate(clips)],
        ensure_ascii=False
    )

    prompt = f"""Pour chacun de ces {len(clips)} clips vidéo, génère en {lang_name} :
- Un titre court et accrocheur (max 8 mots)
- Un hook textuel (phrase d'intro de 1 ligne, max 15 mots, en MAJUSCULES)
- Des hashtags (5 max)

Clips :
{clips_desc}

Réponds en JSON :
{{
  "clips": [
    {{
      "index": 0,
      "title": "...",
      "hook": "...",
      "hashtags": ["#...", "#..."]
    }}
  ]
}}"""

    response = client.messages.create(
        model=MODEL_SONNET,
        max_tokens=1000,
        system=SONNET_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        return clips  # fallback sans titres
    data = json.loads(match.group())

    enriched = dict(enumerate(clips))
    for item in data.get("clips", []):
        idx = item.get("index", 0)
        if idx in enriched:
            enriched[idx]["title"]     = item.get("title", f"Clip {idx+1}")
            enriched[idx]["hook"]      = item.get("hook", "")
            enriched[idx]["hashtags"]  = item.get("hashtags", [])
    return list(enriched.values())

# ── Découpe + sous-titres ─────────────────────────────────────────────────

def cut_clip(video_path: Path, spec: dict, out_dir: Path,
             idx: int, all_segments) -> Path:
    start  = spec["start"]
    end    = spec["end"]
    out    = out_dir / f"{idx}.mp4"
    thumb  = out_dir / f"{idx}_thumb.jpg"

    # Sous-titres pour ce clip (segments chevauchant [start, end])
    subs_path = out_dir / f"{idx}.srt"
    clip_segs = [s for s in all_segments
                 if s.end > start and s.start < end]
    write_srt(clip_segs, start, subs_path)

    # Hook overlay (texte d'intro)
    hook_text = spec.get("hook", "").replace("'", "\\'").replace(":", "\\:")[:60]

    # Filtre ffmpeg : crop vertical 9:16 + sous-titres + hook
    # On centre le crop sur la hauteur, format 9:16
    vf_parts = [
        # Crop 9:16 centré
        "crop=in_h*9/16:in_h:(in_w-in_h*9/16)/2:0",
        # Scale to 1080x1920 (Reels/Shorts/TikTok)
        "scale=1080:1920",
    ]
    if subs_path.exists() and subs_path.stat().st_size > 10:
        vf_parts.append(
            f"subtitles={subs_path}:force_style="
            "'FontName=Arial,FontSize=18,Bold=1,"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "Outline=2,MarginV=80'"
        )

    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out)
    ]
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg échoué : {result.stderr[-300:]}")

    # Thumbnail
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start + 1),
        "-i", str(video_path),
        "-vframes", "1",
        "-vf", "crop=in_h*9/16:in_h:(in_w-in_h*9/16)/2:0,scale=540:960",
        str(thumb)
    ], capture_output=True)

    return out

def write_srt(segments, offset: float, path: Path):
    """Génère un fichier SRT pour les segments donnés."""
    lines = []
    for i, seg in enumerate(segments, 1):
        s = max(0, seg.start - offset)
        e = max(0, seg.end   - offset)
        lines.append(f"{i}")
        lines.append(f"{fmt_time(s)} --> {fmt_time(e)}")
        lines.append(seg.text.strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def fmt_time(secs: float) -> str:
    h  = int(secs // 3600)
    m  = int((secs % 3600) // 60)
    s  = int(secs % 60)
    ms = int((secs - int(secs)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# ── Lancement ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    key = ANTHROPIC_API_KEY
    if not key:
        print("⚠  ANTHROPIC_API_KEY non définie !")
        print("   export ANTHROPIC_API_KEY=sk-ant-...")
    else:
        print(f"✓  Clé Anthropic détectée (…{key[-6:]})")
    print(f"✓  Modèle analyse : {MODEL_OPUS}")
    print(f"✓  Modèle tâches  : {MODEL_SONNET}")
    print()
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
