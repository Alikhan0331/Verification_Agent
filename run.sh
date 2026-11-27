#!/usr/bin/env bash
# Запуск агента одной командой (Linux и macOS):
#
#   bash run.sh
#
# Скрипт сам создаёт окружение .venv, ставит зависимости, при первом запуске спрашивает ключ Gemini
# и запускает проверку документа организаторов. Свой документ и параметры передаются аргументами:
#
#   bash run.sh путь/к/документу.docx отчёт.md --ask
#
# Без аргументов: docs/requirements.md -> report.md.
set -e
cd "$(dirname "$0")"

PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
VPY=.venv/bin/python

# 1. Окружение
if [ ! -x "$VPY" ]; then
    echo "Создаю окружение .venv..."
    rm -rf .venv
    "$PY" -m venv .venv || { echo "Не удалось создать окружение. Нужен Python 3.10 или новее."; exit 1; }
fi

# 2. Зависимости (повторно ставятся только если изменился requirements.txt)
want=$(cksum < requirements.txt)
if [ ! -f .venv/.deps-ok ] || [ "$(cat .venv/.deps-ok)" != "$want" ]; then
    echo "Ставлю зависимости (один раз, около минуты)..."
    "$VPY" -m pip install --quiet --disable-pip-version-check -r requirements.txt
    echo "$want" > .venv/.deps-ok
fi

# 3. Ключ: если его нет ни в окружении, ни в .env, спрашиваем один раз и сохраняем в .env
[ -f .env ] || cp .env.example .env
if [ -z "${GEMINI_API_KEY:-}${GEMINI_API_KEYS:-}${ANTHROPIC_API_KEY:-}" ] \
   && ! grep -Eq '^(GEMINI_API_KEYS?|ANTHROPIC_API_KEY)=[[:space:]]*[^[:space:]]+' <(grep -v 'your_' .env); then
    echo
    echo "Нужен бесплатный ключ Gemini: https://aistudio.google.com/apikey (Create API key)."
    read -r -p "Вставьте ключ и нажмите Enter: " key
    [ -n "$key" ] || { echo "Ключ не введён."; exit 1; }
    "$VPY" - "$key" <<'PYEOF'
import re, sys
key = sys.argv[1].strip()
text = open(".env", encoding="utf-8").read()
text = re.sub(r"(?m)^GEMINI_API_KEY=.*$", lambda m: "GEMINI_API_KEY=" + key, text, count=1)
open(".env", "w", encoding="utf-8", newline="\n").write(text)
PYEOF
    echo "Ключ сохранён в .env."
fi

# 4. Запуск. Без аргументов проверяем документ организаторов; если первым идёт параметр (--ask), добавляем пути.
if [ "$#" -eq 0 ] || [ "${1#-}" != "$1" ]; then
    set -- docs/requirements.md report.md "$@"
fi
exec "$VPY" -m reqcheck "$@"
