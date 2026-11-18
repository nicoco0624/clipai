# ClipAI

Transforme n'importe quelle vidéo YouTube en clips viraux 9:16 (TikTok / Reels / Shorts).

**IA** : Claude Opus 5.5 (analyse) + Claude Sonnet 4.6 (titres/hooks)  
**Video** : yt-dlp + faster-whisper + ffmpeg

---

## Lancement local

```bash
export ANTHROPIC_API_KEY=sk-ant-...
bash start.sh
# → http://localhost:7860
```

## Déploiement

### Render.com (gratuit)

1. Fork ce dépôt sur GitHub
2. Sur [render.com](https://render.com) → **New Web Service** → connecte le dépôt
3. Runtime : **Docker**
4. Dans **Environment** → ajoute `ANTHROPIC_API_KEY`
5. Deploy

### Railway.app

1. Fork le dépôt
2. Sur [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**
3. Ajoute la variable `ANTHROPIC_API_KEY` dans Settings → Variables
4. Deploy

### Hugging Face Spaces

1. Crée un Space (SDK : Docker)
2. Push ce dépôt dans le Space
3. Ajoute `ANTHROPIC_API_KEY` dans Settings → Secrets

---

## Stack

| Composant | Rôle |
|-----------|------|
| FastAPI | Backend API |
| yt-dlp | Téléchargement YouTube |
| faster-whisper | Transcription audio (CPU) |
| Claude Opus 5.5 | Identifier les moments viraux |
| Claude Sonnet 4.6 | Générer titres, hooks, hashtags |
| ffmpeg | Découpe + format 9:16 + sous-titres |
