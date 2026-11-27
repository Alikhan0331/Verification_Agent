# Запуск агента одной командой (Windows PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File .\run.ps1
#
# Скрипт сам создаёт окружение .venv, ставит зависимости, при первом запуске спрашивает ключ Gemini
# и запускает проверку документа организаторов. Свой документ и параметры передаются аргументами:
#
#   powershell -ExecutionPolicy Bypass -File .\run.ps1 путь\к\документу.docx отчёт.md --ask
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

# 3. Ключ. Если его нет ни в окружении, ни в .env, спрашиваем один раз и сохраняем в .env (в репозиторий он не попадает).
$utf8 = New-Object System.Text.UTF8Encoding($false)
$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) { Copy-Item (Join-Path $PSScriptRoot ".env.example") $envFile }
$lines = [System.IO.File]::ReadAllLines($envFile, $utf8)
$fileHasKey = $false
foreach ($l in $lines) {
    if ($l -match '^(GEMINI_API_KEYS?|ANTHROPIC_API_KEY)=\s*(\S+)' -and $Matches[2] -notmatch '^your_') { $fileHasKey = $true }
}
$envHasKey = $env:GEMINI_API_KEY -or $env:GEMINI_API_KEYS -or $env:ANTHROPIC_API_KEY
if (-not $fileHasKey -and -not $envHasKey) {
    Write-Host ""
    Write-Host "Нужен бесплатный ключ Gemini: https://aistudio.google.com/apikey (Create API key)."
    $key = (Read-Host "Вставьте ключ и нажмите Enter").Trim()
    if (-not $key) { Write-Host "Ключ не введён."; exit 1 }
    $new = foreach ($l in $lines) { if ($l -match '^GEMINI_API_KEY=') { "GEMINI_API_KEY=$key" } else { $l } }
    [System.IO.File]::WriteAllLines($envFile, [string[]]$new, $utf8)
    Write-Host "Ключ сохранён в .env."
}

# 4. Запуск. Без аргументов проверяем документ организаторов; если первым идёт параметр (--ask), добавляем пути.
$runArgs = @($args)
if ($runArgs.Count -eq 0 -or $runArgs[0].StartsWith("-")) { $runArgs = @("docs/requirements.md", "report.md") + $runArgs }
& $py -m reqcheck @runArgs
exit $LASTEXITCODE
