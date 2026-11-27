#!/usr/bin/env bash
# Запуск агента одной командой (Linux и macOS):
#
#   bash run.sh
#
# Скрипт сам создаёт окружение .venv, ставит зависимости и запускает проверку документа организаторов.
# Если ключа API нет ни в окружении, ни в .env, агент спросит, какую модель использовать
# (Google Gemini или Anthropic Claude), попросит вставить ключ (ввод скрыт) и сохранит его в .env.
#
# Свой документ и параметры передаются аргументами:
#
#   bash run.sh путь/к/документу.docx отчёт.md --ask
#   bash run.sh --provider anthropic
#
# Без аргументов: docs/requirements.md -> report.md. В живом терминале запуск идёт с --ask: после проверки агент
# сразу задаёт уточняющие вопросы и учитывает ваши ответы. Отключить: --no-ask. Без терминала (пайп, автозапуск)
# вопросов нет, ввода агент не ждёт.
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

# 3. Запуск. Без аргументов проверяем документ организаторов; если первым идёт параметр (--ask), добавляем пути.
if [ "$#" -eq 0 ] || [ "${1#-}" != "$1" ]; then
    set -- docs/requirements.md report.md "$@"
fi
if [ -t 0 ] && [ -t 1 ]; then
    case " $* " in
        *" --ask "*|*" --no-ask "*) ;;
        *) set -- "$@" --ask ;;
    esac
fi
exec "$VPY" -m reqcheck "$@"
