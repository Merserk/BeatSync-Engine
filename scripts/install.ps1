$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$BinDir = Join-Path $Root "bin"
$DownloadsDir = Join-Path $BinDir "downloads"
$UvDir = Join-Path $BinDir "uv"
$PythonVersion = "3.13.17"
$FfmpegVersion = "9.0.2" # Known release when the online latest-version check is unavailable
$LlamaBuild = "b12360"
$LlamaDir = Join-Path $BinDir "llama-bin-win-vulkan-x64"
$PythonDir = Join-Path $BinDir "python-$PythonVersion-embed-amd64"
$PythonExe = Join-Path $PythonDir "python.exe"
$FfmpegDir = Join-Path $BinDir "ffmpeg"
$ModelsDir = Join-Path $BinDir "models"

$PythonZipUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$UvZipUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
$LlamaZipUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaBuild/llama-$LlamaBuild-bin-win-vulkan-x64.zip"
$QwenModelUrl = "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q8_0.gguf?download=true"
$QwenMmprojUrl = "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-F16.gguf?download=true"

function Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Ensure-Dir($Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Assert-InDirectory($Path, $Parent, $Label) {
    $ResolvedPath = (Resolve-Path -LiteralPath $Path).Path
    $ResolvedParent = (Resolve-Path -LiteralPath $Parent).Path
    $IsInside = (
        $ResolvedPath.Equals($ResolvedParent, [System.StringComparison]::OrdinalIgnoreCase) -or
        $ResolvedPath.StartsWith($ResolvedParent + [IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
    )
    if (-not $IsInside) {
        throw "Refusing to operate on $Label outside expected folder: $ResolvedPath"
    }
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

function Download-File($Url, $Path, [long]$MinimumBytes = 1) {
    Ensure-Dir (Split-Path -Parent $Path)

    if (Test-Path $Path) {
        $Existing = Get-Item -LiteralPath $Path
        if ($Existing.Length -ge $MinimumBytes) {
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
    if ((Get-Item -LiteralPath $TempPath).Length -lt $MinimumBytes) {
        Remove-Item -LiteralPath $TempPath -Force -ErrorAction SilentlyContinue
        throw "downloaded file is too small: $Url"
    }

    Move-Item -LiteralPath $TempPath -Destination $Path -Force
}

function Expand-Zip($ZipPath, $Destination) {
    Ensure-Dir $Destination
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $Destination -Force
}

function Remove-SafeFolder($Path, $Parent, $Label) {
    if (-not (Test-Path $Path)) {
        return
    }
    Assert-InDirectory $Path $Parent $Label
    Write-Host "Removing legacy $Label`: $Path"
    Remove-Item -LiteralPath $Path -Recurse -Force
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
    Download-File $UvZipUrl $Archive 1048576
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
    Step "Installing portable Python $PythonVersion"
    if (-not (Test-Path $PythonExe)) {
        $Archive = Join-Path $DownloadsDir "python-$PythonVersion-embed-amd64.zip"
        Download-File $PythonZipUrl $Archive 1048576
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

function Get-LatestFFmpegVersion {
    try {
        $ReleaseVersion = (Invoke-RestMethod -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z.ver' -TimeoutSec 30).Trim()
        if ($ReleaseVersion -notmatch '^\d+\.\d+(?:\.\d+)?$') {
            throw "Unexpected FFmpeg release metadata: $ReleaseVersion"
        }
        return $ReleaseVersion
    } catch {
        Write-Warning "Could not check the latest FFmpeg release; using known release $FfmpegVersion. $($_.Exception.Message)"
        return $FfmpegVersion
    }
}

function Test-FFmpegFullBuild($Directory, $MinimumVersion) {
    try {
        $EncoderExe = Join-Path $Directory 'ffmpeg.exe'
        $ProbeExe = Join-Path $Directory 'ffprobe.exe'
        if (-not (Test-Path $EncoderExe) -or -not (Test-Path $ProbeExe)) { return $false }
        $EncoderVersion = (& $EncoderExe -version 2>$null) -join "`n"
        $ProbeVersion = (& $ProbeExe -version 2>$null) -join "`n"
        $EncoderMatch = [regex]::Match($EncoderVersion, '^ffmpeg version ((\d+\.\d+(?:\.\d+)?)-\S+)')
        $ProbeMatch = [regex]::Match($ProbeVersion, '^ffprobe version (\S+)')
        if (-not $EncoderMatch.Success -or -not $ProbeMatch.Success) { return $false }
        if ($EncoderMatch.Groups[1].Value -ne $ProbeMatch.Groups[1].Value) { return $false }
        if ([version]$EncoderMatch.Groups[2].Value -lt [version]$MinimumVersion) { return $false }
        if ($EncoderVersion -notmatch '--enable-vulkan' -or $EncoderVersion -notmatch '--enable-libplacebo') { return $false }
        $EncoderList = (& $EncoderExe -hide_banner -encoders 2>$null) -join "`n"
        if ($LASTEXITCODE -ne 0) { return $false }
        foreach ($RequiredEncoder in @('libx264','prores_ks','h264_vulkan','hevc_vulkan','h264_nvenc','hevc_nvenc','pcm_s24le','aac')) {
            if ($EncoderList -notmatch ("\b" + [regex]::Escape($RequiredEncoder) + "\b")) { return $false }
        }
        return $true
    } catch { return $false }
}

function Install-FFmpeg {
    $ReleaseVersion = Get-LatestFFmpegVersion
    Step "Checking FFmpeg $ReleaseVersion full build"
    if (Test-FFmpegFullBuild $FfmpegDir $ReleaseVersion) {
        Write-Host "Matching FFmpeg/FFprobe full binaries are ready: $FfmpegDir"
        return
    }
    $AssetName = "ffmpeg-$ReleaseVersion-full_build.zip"
    $Release = Invoke-RestMethod -Uri "https://api.github.com/repos/GyanD/codexffmpeg/releases/tags/$ReleaseVersion" -TimeoutSec 30
    $Asset = $Release.assets | Where-Object { $_.name -eq $AssetName } | Select-Object -First 1
    if (-not $Asset) { throw "FFmpeg full ZIP was not found for release $ReleaseVersion." }
    $Archive = Join-Path $DownloadsDir $AssetName
    Download-File $Asset.browser_download_url $Archive 1048576
    $ArchiveHash = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Asset.digest -and $Asset.digest -match '^sha256:(?<checksum>[a-fA-F0-9]{64})$') {
        if ($ArchiveHash -ne $Matches.checksum.ToLowerInvariant()) {
            throw "FFmpeg archive checksum mismatch. Remove $Archive and run the installer again."
        }
    }
    $ExtractDir = Join-Path $DownloadsDir 'ffmpeg-extract'
    if (Test-Path $ExtractDir) {
        Assert-InDirectory $ExtractDir $DownloadsDir 'FFmpeg extraction cache'
        Remove-Item -LiteralPath $ExtractDir -Recurse -Force
    }
    Expand-Zip $Archive $ExtractDir
    $ExtractedExe = Get-ChildItem -LiteralPath $ExtractDir -Recurse -Filter 'ffmpeg.exe' | Select-Object -First 1
    if (-not $ExtractedExe) { throw 'The FFmpeg archive does not contain ffmpeg.exe.' }
    $ExtractedBin = Split-Path -Parent $ExtractedExe.FullName
    if (-not (Test-FFmpegFullBuild $ExtractedBin $ReleaseVersion)) {
        throw 'The downloaded FFmpeg full build failed version/capability validation.'
    }
    Ensure-Dir $FfmpegDir
    foreach ($Binary in @('ffmpeg.exe','ffprobe.exe','ffplay.exe')) {
        $SourceBinary = Join-Path $ExtractedBin $Binary
        if (Test-Path $SourceBinary) { Copy-Item -LiteralPath $SourceBinary -Destination $FfmpegDir -Force }
    }
    if (-not (Test-FFmpegFullBuild $FfmpegDir $ReleaseVersion)) {
        throw 'The installed FFmpeg/FFprobe pair failed validation.'
    }
    @{ version = $ReleaseVersion; build = 'full'; source = $Asset.browser_download_url; archive_sha256 = $ArchiveHash } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $FfmpegDir 'beatsync-runtime.json') -Encoding UTF8
}

function Install-LlamaCppVulkan {
    Step "Installing llama.cpp Vulkan $LlamaBuild"
    $ServerExe = Join-Path $LlamaDir "llama-server.exe"
    $MtmdExe = Join-Path $LlamaDir "llama-mtmd-cli.exe"
    $CliExe = Join-Path $LlamaDir "llama-cli.exe"
    $ExpectedLlamaBuildNumber = $LlamaBuild -replace '^b', ''
    if ((Test-Path $ServerExe) -and (Test-Path $MtmdExe) -and (Test-Path $CliExe)) {
        try {
            $CurrentVersion = (& $CliExe --version 2>$null | Select-Object -First 1)
            if ($CurrentVersion -match "(?:version:\s+|build\s+)$([regex]::Escape($ExpectedLlamaBuildNumber))\b") {
                Write-Host "llama.cpp Vulkan ready: $LlamaDir"
                return
            }
            Write-Host "Existing llama.cpp build is not $LlamaBuild; replacing Vulkan binaries."
        } catch {
            Write-Host "Existing llama.cpp version check failed; replacing Vulkan binaries."
        }
    }

    $Archive = Join-Path $DownloadsDir "llama-$LlamaBuild-bin-win-vulkan-x64.zip"
    Download-File $LlamaZipUrl $Archive 1048576
    $ExtractDir = Join-Path $DownloadsDir "llama-vulkan-extract"
    if (Test-Path $ExtractDir) {
        Remove-Item -LiteralPath $ExtractDir -Recurse -Force
    }
    Expand-Zip $Archive $ExtractDir

    $ExtractedServer = Get-ChildItem -Path $ExtractDir -Recurse -Filter "llama-server.exe" | Select-Object -First 1
    if (-not $ExtractedServer) {
        throw "llama.cpp archive extracted, but llama-server.exe was not found."
    }

    if (Test-Path $LlamaDir) {
        Remove-Item -LiteralPath $LlamaDir -Recurse -Force
    }
    Ensure-Dir $LlamaDir
    $ExtractedBin = Split-Path -Parent $ExtractedServer.FullName
    Copy-Item -Path (Join-Path $ExtractedBin "*") -Destination $LlamaDir -Recurse -Force
}

function Install-PythonPackages($UvExe) {
    Step "Installing Python packages with UV"
    $Env:UV_LINK_MODE = "copy"
    $Env:PATH = @(
        $FfmpegDir,
        $PythonDir,
        (Join-Path $PythonDir "Scripts"),
        $Env:PATH
    ) -join [IO.Path]::PathSeparator

    & $UvExe pip install --python $PythonExe --system --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw "UV failed while installing pip/setuptools/wheel." }

    & $UvExe pip install --python $PythonExe --system --upgrade -r (Join-Path $Root "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "UV failed while installing app requirements." }
}

function Remove-LegacyPythonPackages($UvExe) {
    Step "Removing superseded Python GPU packages"
    $Packages = @(
        "torch", "torchvision", "torchaudio", "accelerate", "transformers", "safetensors",
        "cupy", "cupy-cuda11x", "cupy-cuda12x", "cupy-cuda13x", "cuda-toolkit", "cuda-pathfinder",
        "nvidia-cublas", "nvidia-cuda-nvrtc", "nvidia-cuda-runtime", "nvidia-cufft",
        "nvidia-curand", "nvidia-cusolver", "nvidia-cusparse", "nvidia-nvjitlink"
    )
    & $UvExe pip uninstall --python $PythonExe --system -y @Packages
    if ($LASTEXITCODE -ne 0) {
        throw "UV failed while removing superseded Python GPU packages."
    }
}

function Install-QwenGgufModels {
    Step "Installing Qwen3.5 GGUF models (thinking off)"
    Ensure-Dir $ModelsDir
    Download-File $QwenModelUrl (Join-Path $ModelsDir "Qwen3.5-2B-Q8_0.gguf") 104857600
    Download-File $QwenMmprojUrl (Join-Path $ModelsDir "mmproj-F16.gguf") 104857600
}

function Remove-LegacyQwenModels {
    Step "Removing superseded Qwen3-VL GGUF models"
    foreach ($Name in @("Qwen3VL-2B-Instruct-Q8_0.gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf")) {
        $LegacyModel = Join-Path $ModelsDir $Name
        if (Test-Path -LiteralPath $LegacyModel) {
            Assert-InDirectory $LegacyModel $ModelsDir "legacy Qwen model"
            Remove-Item -LiteralPath $LegacyModel -Force
        }
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

function Remove-LegacyFiles {
    Step "Removing legacy runtime files"
    Get-ChildItem -LiteralPath $BinDir -Directory -Filter "python-*-embed-amd64" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -ne $PythonDir } |
        ForEach-Object { Remove-SafeFolder $_.FullName $BinDir "legacy portable Python ($($_.Name))" }
    Remove-SafeFolder (Join-Path $BinDir "PortableGit") $BinDir "PortableGit folder"
    Remove-SafeFolder (Join-Path $BinDir "CUDA\v13.3") $BinDir "portable CUDA Toolkit"
    Remove-SafeFolder (Join-Path $BinDir "models\Qwen3-VL-2B-Instruct") $BinDir "Transformers Qwen model"

    $LegacyMinGitZip = Join-Path $DownloadsDir "MinGit-2.54.0-64-bit.zip"
    if (Test-Path $LegacyMinGitZip) {
        Remove-Item -LiteralPath $LegacyMinGitZip -Force
    }

    $CudaRoot = Join-Path $BinDir "CUDA"
    if ((Test-Path $CudaRoot) -and (-not (Get-ChildItem -LiteralPath $CudaRoot -Force -ErrorAction SilentlyContinue))) {
        Remove-Item -LiteralPath $CudaRoot -Force
    }
}

function Remove-InstallerFolder($Path, $Label) {
    if (-not (Test-Path $Path)) {
        return
    }
    Assert-InDirectory $Path $BinDir $Label
    Write-Host "Cleaning $Label`: $Path"
    Remove-Item -LiteralPath $Path -Recurse -Force
}

function Cleanup-InstallerFiles {
    Step "Cleaning installer cache"
    Remove-InstallerFolder $DownloadsDir "downloads"
    Remove-InstallerFolder $UvDir "UV"
}

function Test-RequiredFile($Path, $Label) {
    if (-not (Test-Path $Path)) {
        throw "Missing $Label`: $Path"
    }
}

Ensure-Dir $BinDir
Ensure-Dir $DownloadsDir
Ensure-AppFolders
Remove-LegacyFiles

Install-Python
$UvExe = Install-Uv
Install-FFmpeg
Install-LlamaCppVulkan
Install-PythonPackages $UvExe
Remove-LegacyPythonPackages $UvExe
Install-QwenGgufModels

Step "Verifying portable install"
& $PythonExe -X utf8 -c "from ffmpeg_runtime import get_runtime_info; r=get_runtime_info(); print('FFmpeg', r['version'], 'full build; matching FFprobe; required encoders verified')"
if ($LASTEXITCODE -ne 0) {
    throw "Portable FFmpeg full-build verification failed."
}
& $PythonExe -X utf8 -c "import sys, gradio, librosa, cv2, numpy, taichi, numba; from gpu_cpu_utils import GPU_INFO; print('Python', sys.version.split()[0]); print('gradio', gradio.__version__); print('librosa', librosa.__version__); print('Taichi', taichi.__version__); print('numba', numba.__version__); print('GPU analysis', GPU_INFO); print('CPU fallback' if not GPU_INFO['available'] else 'Vulkan compute verified')"
if ($LASTEXITCODE -ne 0) {
    throw "Portable app import/Taichi verification failed."
}

& $PythonExe -X utf8 -c "import importlib.util, sys; missing = [name for name in ('torch', 'torchvision', 'torchaudio', 'accelerate', 'transformers', 'safetensors', 'cupy') if importlib.util.find_spec(name) is not None]; print('legacy packages', missing); sys.exit(1 if missing else 0)"
if ($LASTEXITCODE -ne 0) {
    throw "Superseded Python GPU packages are still installed."
}

Test-RequiredFile (Join-Path $LlamaDir "llama-server.exe") "llama-server.exe"
Test-RequiredFile (Join-Path $LlamaDir "llama-mtmd-cli.exe") "llama-mtmd-cli.exe"
Test-RequiredFile (Join-Path $LlamaDir "llama-cli.exe") "llama-cli.exe"
Test-RequiredFile (Join-Path $ModelsDir "Qwen3.5-2B-Q8_0.gguf") "Qwen GGUF model"
Test-RequiredFile (Join-Path $ModelsDir "mmproj-F16.gguf") "Qwen mmproj model"

& (Join-Path $LlamaDir "llama-cli.exe") --version
if ($LASTEXITCODE -ne 0) {
    throw "llama-cli.exe --version failed."
}

& $PythonExe -X utf8 -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "pip check failed."
}

Remove-LegacyQwenModels
Cleanup-InstallerFiles

Write-Host ""
Write-Host "Portable install is ready: llama.cpp Vulkan + Taichi Vulkan, no PyTorch." -ForegroundColor Green
