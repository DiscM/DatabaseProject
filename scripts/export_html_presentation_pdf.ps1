param(
    [string]$HtmlPath = "output\presentations\oil_news_project_presentation.html",
    [string]$PdfPath = "output\presentations\oil_news_project_presentation_from_html.pdf",
    [string]$ExpectedText = "Semantic News and Oil Price Database Project",
    [int]$ExpectedPages = 9
)

$ErrorActionPreference = "Stop"

function Resolve-Browser {
    $candidates = @(
        "C:\Program Files\Google\Chrome\Application\chrome.exe",
        "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    throw "Chrome or Edge was not found. Install one of them or update Resolve-Browser in this script."
}

function Resolve-Python {
    $bundled = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path -LiteralPath $bundled) {
        return (Resolve-Path -LiteralPath $bundled).Path
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }

    throw "Python was not found. It is only used to verify the exported PDF."
}

$projectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$html = Resolve-Path -LiteralPath (Join-Path $projectRoot $HtmlPath)
$pdf = Join-Path $projectRoot $PdfPath
$pdfDir = Split-Path -Parent $pdf
$profile = Join-Path $projectRoot "tmp\chrome-html-pdf-profile"

New-Item -ItemType Directory -Force -Path $pdfDir | Out-Null
New-Item -ItemType Directory -Force -Path $profile | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $profile "Crashpad") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $profile "Crashpad\attachments") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $profile "Crashpad\reports") | Out-Null

$browser = Resolve-Browser
$fileUrl = "file:///" + ($html.Path -replace "\\", "/")
$args = @(
    "--headless",
    "--disable-gpu",
    "--disable-extensions",
    "--disable-crash-reporter",
    "--disable-crashpad",
    "--no-first-run",
    "--no-default-browser-check",
    "--user-data-dir=$profile",
    "--print-to-pdf-no-header",
    "--print-to-pdf=$pdf",
    $fileUrl
)

try {
    if (Test-Path -LiteralPath $pdf) {
        Remove-Item -LiteralPath $pdf -Force
    }

    $process = Start-Process -FilePath $browser -ArgumentList $args -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0 -and -not (Test-Path -LiteralPath $pdf)) {
        throw "Browser export failed with exit code $($process.ExitCode)."
    }

    if (-not (Test-Path -LiteralPath $pdf)) {
        throw "Browser completed but did not create $pdf."
    }

    $python = Resolve-Python
    $verifyScript = @"
from pathlib import Path
from pypdf import PdfReader

pdf_path = Path(r"$pdf")
expected_text = r"$ExpectedText"
expected_pages = $ExpectedPages

reader = PdfReader(str(pdf_path))
text = "\n".join((page.extract_text() or "") for page in reader.pages)

if len(reader.pages) != expected_pages:
    raise SystemExit(f"Expected {expected_pages} pages, found {len(reader.pages)}")
if expected_text not in text:
    raise SystemExit(f"Expected text not found: {expected_text}")
if "1. Cover 2. Workflow" in text:
    raise SystemExit("Navigation text appears in the PDF; print CSS may not be hiding the sidebar.")

print(f"Verified {pdf_path} ({len(reader.pages)} pages, {pdf_path.stat().st_size} bytes)")
"@

    $verifyScript | & $python -
    Write-Host "Exported HTML presentation PDF: $pdf"
}
finally {
    if (Test-Path -LiteralPath $profile) {
        Remove-Item -LiteralPath $profile -Recurse -Force
    }
}
