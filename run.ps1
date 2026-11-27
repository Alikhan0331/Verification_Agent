﻿# Запуск агента одной командой (Windows PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File .\run.ps1
#
# Скрипт сам создаёт окружение .venv, ставит зависимости и запускает проверку документа организаторов.
# Если ключа API нет ни в окружении, ни в .env, агент спросит, какую модель использовать
# (Google Gemini или Anthropic Claude), попросит вставить ключ (ввод скрыт) и сохранит его в .env.
#
# Свой документ и параметры передаются аргументами:
#
#   powershell -ExecutionPolicy Bypass -File .\run.ps1 путь\к\документу.docx отчёт.md --ask
#   powershell -ExecutionPolicy Bypass -File .\run.ps1 --provider anthropic
#
# Без аргументов: docs/requirements.md -> report.md.

Set-Location $PSScriptRoot
$venv = Join-Path $PSScriptRoot ".venv"
$py = Join-Path $venv "Scripts\python.exe"

# 1. Окружение. Если в .venv нет Scripts, его создал не тот Python (например, из MSYS2): пересоздаём.
if ((Test-Path $venv) -and -not (Test-Path $py)) {
    Write-Host "Окружение .venv создано другим Python, пересоздаю..."
    Remove-Item -Recurse -Force $venv
}
if (-not (Test-Path $py)) {
    Write-Host "Создаю окружение .venv..."
    if (Get-Command py -ErrorAction SilentlyContinue) { & py -3 -m venv $venv } else { & python -m venv $venv }
    if (-not (Test-Path $py)) {
        Write-Host "Не удалось создать окружение. Установите Python 3.10 или новее с https://www.python.org/downloads/ и повторите."
        exit 1
    }
}

# 2. Зависимости (повторно ставятся только если изменился requirements.txt).
$stamp = Join-Path $venv ".deps-ok"
$want = (Get-FileHash (Join-Path $PSScriptRoot "requirements.txt") -Algorithm SHA256).Hash
if (-not (Test-Path $stamp) -or ((Get-Content $stamp -Raw).Trim() -ne $want)) {
    Write-Host "Ставлю зависимости (один раз, около минуты)..."
    & $py -m pip install --quiet --disable-pip-version-check -r (Join-Path $PSScriptRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { Write-Host "Не удалось поставить зависимости."; exit 1 }
    Set-Content -Path $stamp -Value $want
}

# 3. Запуск. Без аргументов проверяем документ организаторов; если первым идёт параметр (--ask), добавляем пути.
$runArgs = @($args)
if ($runArgs.Count -eq 0 -or $runArgs[0].StartsWith("-")) { $runArgs = @("docs/requirements.md", "report.md") + $runArgs }
& $py -m reqcheck @runArgs
exit $LASTEXITCODE
