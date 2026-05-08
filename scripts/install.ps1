$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$BinDir = Join-Path $Root "bin"
$DownloadsDir = Join-Path $BinDir "downloads"
$UvDir = Join-Path $BinDir "uv"
$PythonDir = Join-Path $BinDir "python-3.13.13-embed-amd64"
$PythonExe = Join-Path $PythonDir "python.exe"
$CudaDir = Join-Path $BinDir "CUDA\v13.0"
$FfmpegDir = Join-Path $BinDir "ffmpeg"
$ModelDir = Join-Path $BinDir "models\Qwen3-VL-2B-Instruct"

$PythonZipUrl = "https://www.python.org/ftp/python/3.13.13/python-3.13.13-embed-amd64.zip"
$UvZipUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
$FfmpegZipUrl = "https://github.com/GyanD/codexffmpeg/releases/download/8.1.1/ffmpeg-8.1.1-essentials_build.zip"
$CudaManifestUrl = "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.2.json"
$CudaBaseUrl = "https://developer.download.nvidia.com/compute/cuda/redist/"
$TorchIndexUrl = "https://download.pytorch.org/whl/cu130"

function Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Ensure-Dir($Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Get-CurlExe {
    $SystemCurl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($SystemCurl) {
        return $SystemCurl.Source
    }
    throw "curl.exe was not found. Windows 10 1803+ includes curl.exe; install curl or update Windows."
}

$script:CurlHelpText = $null
function Test-CurlOption($CurlExe, $Option) {
    if ($null -eq $script:CurlHelpText) {
        try {
            $script:CurlHelpText = (& $CurlExe --help all 2>$null) -join "`n"
        } catch {
            $script:CurlHelpText = ""
        }
    }
    return ($script:CurlHelpText -match [regex]::Escape($Option))
}

function Invoke-CurlDownload($CurlExe, $Url, $OutFile, [bool]$Resume) {
    $FailOption = "--fail"
    if (Test-CurlOption $CurlExe "--fail-with-body") {
        $FailOption = "--fail-with-body"
    }

    $CurlArgs = @(
        "--location",
        $FailOption,
        "--show-error",
        "--retry", "12",
        "--retry-delay", "2",
        "--retry-max-time", "0",
        "--connect-timeout", "30",
        "--speed-time", "60",
        "--speed-limit", "1024",
        "--user-agent", "BeatSync-Installer/1.0",
        "--output", $OutFile
    )

    if (Test-CurlOption $CurlExe "--retry-all-errors") {
        $CurlArgs += "--retry-all-errors"
    }
    if (Test-CurlOption $CurlExe "--retry-connrefused") {
        $CurlArgs += "--retry-connrefused"
    }
    if (Test-CurlOption $CurlExe "--tcp-nodelay") {
        $CurlArgs += "--tcp-nodelay"
    }
    if ($Resume) {
        $CurlArgs += @("--continue-at", "-")
    }

    $CurlArgs += $Url
    & $CurlExe @CurlArgs
    return $LASTEXITCODE
}

function Download-File($Url, $Path) {
    Ensure-Dir (Split-Path -Parent $Path)

    if (Test-Path $Path) {
        $Existing = Get-Item -LiteralPath $Path
        if ($Existing.Length -gt 0) {
            Write-Host "Using cached file: $Path"
            return
        }
        Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    }

    $TempPath = "$Path.partial"
    $CurlExe = Get-CurlExe
    Write-Host "Downloading: $Url"
    Write-Host "Using curl: $CurlExe"

    $Resume = $false
    if (Test-Path $TempPath) {
        $Partial = Get-Item -LiteralPath $TempPath
        if ($Partial.Length -gt 0) {
            $Resume = $true
            Write-Host "Resuming partial file: $TempPath"
        } else {
            Remove-Item -LiteralPath $TempPath -Force -ErrorAction SilentlyContinue
        }
    }

    $ExitCode = Invoke-CurlDownload $CurlExe $Url $TempPath $Resume
    if (($ExitCode -ne 0) -and $Resume) {
        Write-Host "Resume failed; retrying once from scratch." -ForegroundColor Yellow
        Remove-Item -LiteralPath $TempPath -Force -ErrorAction SilentlyContinue
        $ExitCode = Invoke-CurlDownload $CurlExe $Url $TempPath $false
    }

    if ($ExitCode -ne 0) {
        throw "curl download failed with exit code $ExitCode`: $Url"
    }
    if (-not (Test-Path $TempPath)) {
        throw "curl reported success, but output file was not created: $TempPath"
    }
    if ((Get-Item -LiteralPath $TempPath).Length -le 0) {
        Remove-Item -LiteralPath $TempPath -Force -ErrorAction SilentlyContinue
        throw "downloaded file is empty: $Url"
    }

    Move-Item -LiteralPath $TempPath -Destination $Path -Force
}

function Expand-Zip($ZipPath, $Destination) {
    Ensure-Dir $Destination
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $Destination -Force
}

function Install-Uv {
    Step "Preparing UV"
    $UvExe = Join-Path $UvDir "uv.exe"
    if (Test-Path $UvExe) {
        Write-Host "UV ready: $UvExe"
        return $UvExe
    }

    Ensure-Dir $UvDir
    $Archive = Join-Path $DownloadsDir "uv-x86_64-pc-windows-msvc.zip"
    Download-File $UvZipUrl $Archive
    Expand-Zip $Archive $UvDir

    $Found = Get-ChildItem -Path $UvDir -Recurse -Filter "uv.exe" | Select-Object -First 1
    if (-not $Found) {
        throw "UV archive extracted, but uv.exe was not found."
    }
    if ($Found.FullName -ne $UvExe) {
        Copy-Item -LiteralPath $Found.FullName -Destination $UvExe -Force
    }
    return $UvExe
}

function Install-Python {
    Step "Installing portable Python 3.13.13"
    if (-not (Test-Path $PythonExe)) {
        $Archive = Join-Path $DownloadsDir "python-3.13.13-embed-amd64.zip"
        Download-File $PythonZipUrl $Archive
        if (Test-Path $PythonDir) {
            Remove-Item -LiteralPath $PythonDir -Recurse -Force
        }
        Expand-Zip $Archive $PythonDir
    }

    $PthFile = Join-Path $PythonDir "python313._pth"
    @(
        "python313.zip",
        ".",
        "Lib\site-packages",
        "..\..\src",
        "import site"
    ) | Set-Content -LiteralPath $PthFile -Encoding ASCII

    Ensure-Dir (Join-Path $PythonDir "Lib\site-packages")
    Write-Host "Python ready: $PythonExe"
}

function Install-FFmpeg {
    Step "Installing FFmpeg release zip"
    $FfmpegExe = Join-Path $FfmpegDir "ffmpeg.exe"
    $FfprobeExe = Join-Path $FfmpegDir "ffprobe.exe"
    if ((Test-Path $FfmpegExe) -and (Test-Path $FfprobeExe)) {
        Write-Host "FFmpeg ready: $FfmpegDir"
        return
    }

    Ensure-Dir $FfmpegDir
    $Archive = Join-Path $DownloadsDir "ffmpeg-8.1.1-essentials_build.zip"
    Download-File $FfmpegZipUrl $Archive
    $ExtractDir = Join-Path $DownloadsDir "ffmpeg-extract"
    if (Test-Path $ExtractDir) {
        Remove-Item -LiteralPath $ExtractDir -Recurse -Force
    }
    Expand-Zip $Archive $ExtractDir

    $ExtractedFfmpeg = Get-ChildItem -Path $ExtractDir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
    if (-not $ExtractedFfmpeg) {
        throw "FFmpeg archive extracted, but ffmpeg.exe was not found."
    }
    $ExtractedBin = Split-Path -Parent $ExtractedFfmpeg.FullName
    Copy-Item -LiteralPath (Join-Path $ExtractedBin "ffmpeg.exe") -Destination $FfmpegDir -Force
    Copy-Item -LiteralPath (Join-Path $ExtractedBin "ffprobe.exe") -Destination $FfmpegDir -Force
}

function Copy-CudaArchiveContent($ExtractDir) {
    foreach ($Name in @("bin", "include", "lib")) {
        $Folders = Get-ChildItem -Path $ExtractDir -Directory -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -eq $Name }
        foreach ($Folder in $Folders) {
            $Target = Join-Path $CudaDir $Name
            Ensure-Dir $Target
            Copy-Item -Path (Join-Path $Folder.FullName "*") -Destination $Target -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    $Licenses = Get-ChildItem -Path $ExtractDir -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "^(LICENSE|EULA|version\.json)" }
    foreach ($File in $Licenses) {
        Copy-Item -LiteralPath $File.FullName -Destination $CudaDir -Force -ErrorAction SilentlyContinue
    }
}

function Install-CudaRedistributables {
    Step "Installing portable CUDA 13.0 redistributable components"
    $Cudart = Get-ChildItem -Path $CudaDir -Recurse -Filter "cudart64_13.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
    $Nvrtc = Get-ChildItem -Path $CudaDir -Recurse -Filter "nvrtc64_*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($Cudart -and $Nvrtc) {
        Write-Host "CUDA redistributables already present: $CudaDir"
        return
    }

    Ensure-Dir $CudaDir
    $ManifestPath = Join-Path $DownloadsDir "redistrib_13.0.2.json"
    Download-File $CudaManifestUrl $ManifestPath
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    $Components = @(
        "cuda_cudart",
        "cuda_nvrtc",
        "cuda_nvtx",
        "cuda_opencl",
        "cuda_nvml_dev",
        "libcublas",
        "libcufft",
        "libcurand",
        "libcusolver",
        "libcusparse",
        "libnpp",
        "libnvfatbin",
        "libnvjitlink",
        "libnvjpeg",
        "libnvptxcompiler"
    )

    foreach ($Component in $Components) {
        $Info = $Manifest.$Component
        if (-not $Info) {
            Write-Host "Skipping missing CUDA manifest component: $Component"
            continue
        }
        $Package = $Info."windows-x86_64"
        if (-not $Package) {
            Write-Host "Skipping CUDA component without Windows package: $Component"
            continue
        }

        $Url = $CudaBaseUrl + $Package.relative_path
        $Archive = Join-Path $DownloadsDir (Split-Path -Leaf $Package.relative_path)
        Download-File $Url $Archive

        $ExtractDir = Join-Path $DownloadsDir ("cuda-" + $Component)
        if (Test-Path $ExtractDir) {
            Remove-Item -LiteralPath $ExtractDir -Recurse -Force
        }
        Expand-Zip $Archive $ExtractDir
        Copy-CudaArchiveContent $ExtractDir
    }

    $VersionPath = Join-Path $CudaDir "version.json"
    @{
        cuda = @{
            name = "CUDA redistributables"
            version = $Manifest.release_label
        }
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $VersionPath -Encoding UTF8
}

function Install-PythonPackages($UvExe) {
    Step "Installing Python packages with UV"
    $Env:UV_LINK_MODE = "copy"
    $Env:CUDA_PATH = $CudaDir
    $Env:CUDA_HOME = $CudaDir
    $Env:CUDA_ROOT = $CudaDir
    $Env:PATH = @(
        (Join-Path $CudaDir "bin\x64"),
        (Join-Path $CudaDir "bin"),
        (Join-Path $CudaDir "lib\x64"),
        $FfmpegDir,
        $PythonDir,
        (Join-Path $PythonDir "Scripts"),
        $Env:PATH
    ) -join [IO.Path]::PathSeparator

    & $UvExe pip install --python $PythonExe --system --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw "UV failed while installing pip/setuptools/wheel." }

    & $UvExe pip install --python $PythonExe --system `
        "torch==2.11.0+cu130" `
        "torchvision==0.26.0+cu130" `
        "torchaudio==2.11.0+cu130" `
        --index-url $TorchIndexUrl
    if ($LASTEXITCODE -ne 0) { throw "UV failed while installing PyTorch CUDA packages." }

    & $UvExe pip install --python $PythonExe --system --index-url "https://pypi.org/simple" "setuptools>=82.0.1" wheel
    if ($LASTEXITCODE -ne 0) { throw "UV failed while restoring packaging tools after PyTorch install." }

    & $UvExe pip install --python $PythonExe --system -r (Join-Path $Root "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "UV failed while installing app requirements." }
}

function Install-QwenModel {
    Step "Installing Qwen3-VL-2B-Instruct model"
    $ModelWeights = Join-Path $ModelDir "model.safetensors"
    if ((Test-Path $ModelDir) -and
        (Test-Path (Join-Path $ModelDir "config.json")) -and
        (Test-Path $ModelWeights) -and
        ((Get-Item -LiteralPath $ModelWeights).Length -gt 1048576)) {
        Write-Host "Model already present: $ModelDir"
        return
    }

    Ensure-Dir (Split-Path -Parent $ModelDir)
    if (Test-Path $ModelDir) {
        Remove-Item -LiteralPath $ModelDir -Recurse -Force
    }

    $DownloadScript = Join-Path $DownloadsDir "download_qwen_model.py"
    $Code = @"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-VL-2B-Instruct',
    local_dir=r'''$ModelDir''',
    max_workers=8,
)
"@
    Set-Content -LiteralPath $DownloadScript -Value $Code -Encoding UTF8
    & $PythonExe -X utf8 $DownloadScript
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to download Qwen/Qwen3-VL-2B-Instruct."
    }
}

function Ensure-AppFolders {
    Step "Creating app folders"
    foreach ($Path in @(
        "input",
        "input\audio",
        "input\video",
        "input\processing",
        "input\gradio_uploads",
        "input\video_analysis_cache",
        "output"
    )) {
        Ensure-Dir (Join-Path $Root $Path)
    }
}

function Remove-LegacyGitFiles {
    $LegacyGitDir = Join-Path $BinDir "PortableGit"
    $LegacyMinGitZip = Join-Path $DownloadsDir "MinGit-2.54.0-64-bit.zip"
    if (Test-Path $LegacyGitDir) {
        Step "Removing legacy portable Git folder"
        Remove-Item -LiteralPath $LegacyGitDir -Recurse -Force
    }
    if (Test-Path $LegacyMinGitZip) {
        Remove-Item -LiteralPath $LegacyMinGitZip -Force
    }
}

function Remove-InstallerFolder($Path, $Label) {
    if (-not (Test-Path $Path)) {
        return
    }
    $ResolvedPath = (Resolve-Path -LiteralPath $Path).Path
    $ResolvedBin = (Resolve-Path -LiteralPath $BinDir).Path
    if (-not $ResolvedPath.StartsWith($ResolvedBin, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean $Label outside bin folder: $ResolvedPath"
    }
    Write-Host "Cleaning $Label`: $ResolvedPath"
    Remove-Item -LiteralPath $ResolvedPath -Recurse -Force
}

function Cleanup-InstallerFiles {
    Step "Cleaning installer cache"
    Remove-InstallerFolder $DownloadsDir "downloads"
    Remove-InstallerFolder $UvDir "UV"
}

Ensure-Dir $BinDir
Ensure-Dir $DownloadsDir
Ensure-AppFolders
Remove-LegacyGitFiles

Install-Python
$UvExe = Install-Uv
Install-FFmpeg
Install-CudaRedistributables
Install-PythonPackages $UvExe
Install-QwenModel

Step "Verifying portable install"
& $PythonExe -X utf8 -c "import sys, gradio, librosa, cv2, numpy, cupy; print('Python', sys.version.split()[0]); print('gradio', gradio.__version__); print('librosa', librosa.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "Portable app import verification failed."
}

& $PythonExe -X utf8 -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "Portable Torch/Transformers import verification failed."
}

Cleanup-InstallerFiles

Write-Host ""
Write-Host "Portable install is ready." -ForegroundColor Green
