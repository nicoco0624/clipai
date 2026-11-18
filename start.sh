#!/usr/bin/env bash
# Lance ClipAI sur http://localhost:7860
set -euo pipefail

CLIPAI_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$CLIPAI_DIR/.venv"

# Créer le venv si nécessaire
if [[ ! -d "$VENV" ]]; then
    echo "→ Création du venv Python…"
    python3 -m venv "$VENV"
fi

# Installer les dépendances
echo "→ Vérification des dépendances…"
"$VENV/bin/pip" install --quiet -r "$CLIPAI_DIR/requirements.txt"

# Vérifier la clé API
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo ""
    echo "⚠  ANTHROPIC_API_KEY non définie !"
    echo "   Exécute : export ANTHROPIC_API_KEY=sk-ant-..."
    echo "   Puis relance ce script."
    echo ""
    exit 1
fi

echo ""
echo "✓  ClipAI démarre sur http://localhost:7860"
echo "   Ctrl+C pour arrêter."
echo ""

cd "$CLIPAI_DIR"
"$VENV/bin/python3" main.py
