# chordgen ポータブル版: Python と Node.js をアプリのフォルダ（runtime\）に置く。
# run.bat から呼ばれる。版・URL・sha256 は portable\sources.json。冪等で、途中で失敗しても再実行で続きから進む。
# Windows PowerShell 5.1 で動くように書く（UTF-8 BOM 付きで保存する。BOM がないと 5.1 は日本語を読み違える）。

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # 進捗表示があると Invoke-WebRequest が極端に遅くなる
# 古い .NET の既定は TLS 1.0 のことがある（このプロセスの中だけで変える）
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$App = Split-Path -Parent $PSScriptRoot
$Runtime = Join-Path $App 'runtime'
$Downloads = Join-Path $Runtime 'downloads'
$Sources = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $PSScriptRoot 'sources.json') | ConvertFrom-Json

function Fail([string]$Message) {
    Write-Host ''
    Write-Host "[chordgen] $Message" -ForegroundColor Red
    exit 1
}

# 外部コマンドを黙って走らせて成功したかを返す。5.1 は Stop のまま標準エラーをリダイレクトすると例外にするので Continue にする
function Test-Native([scriptblock]$Block) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Block *> $null; return ($LASTEXITCODE -eq 0) }
    catch { return $false }
    finally { $ErrorActionPreference = $old }
}

function Get-Sha256([string]$Path) {
    (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

# URL のファイルを取得して sha256 を確かめる。済んでいれば何もしない。curl.exe があれば途中から再開する。
function Get-Verified([string]$Url, [string]$Sha256, [string]$Dest) {
    if ((Test-Path -LiteralPath $Dest) -and ((Get-Sha256 $Dest) -eq $Sha256)) { return }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Dest) | Out-Null
    $part = "$Dest.part"
    $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
    Write-Host "  取得: $Url"
    if (Test-Path -LiteralPath $curl) {
        & $curl --fail --location --retry 3 --continue-at - --output $part $Url
        if ($LASTEXITCODE -ne 0) {
            # 再開できない（サーバーが範囲指定に対応しない・壊れた .part）ときは最初から
            Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
            & $curl --fail --location --retry 3 --output $part $Url
            if ($LASTEXITCODE -ne 0) { Fail "取得に失敗しました（curl の終了コード $LASTEXITCODE）: $Url`nネットにつながっているか、URL が今もあるかを確かめてください。" }
        }
    } else {
        try { Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $part }
        catch { Fail "取得に失敗しました: $Url`n$($_.Exception.Message)" }
    }
    $got = Get-Sha256 $part
    if ($got -ne $Sha256) {
        Remove-Item -LiteralPath $part -Force
        Fail "取得したファイルの sha256 が合いません: $Url`n  期待 $Sha256`n  実際 $got`nもう一度実行してください。続くなら portable\sources.json の版を確かめてください。"
    }
    Move-Item -LiteralPath $part -Destination $Dest -Force
}

# アーカイブを一時フォルダに展開し、中の $Root フォルダを $Target に置く。印（.chordgen-source）が合えば飛ばす。
function Install-Archive([string]$Name, $Spec, [string]$Target) {
    $marker = Join-Path $Target '.chordgen-source'
    if ((Test-Path -LiteralPath $marker) -and ((Get-Content -Raw -LiteralPath $marker).Trim() -eq $Spec.sha256)) {
        Write-Host "  $Name $($Spec.version) は入っています"
        return
    }
    $fileName = [IO.Path]::GetFileName(([Uri]$Spec.url).AbsolutePath) -replace '%2B', '+'
    $archive = Join-Path $Downloads $fileName
    Get-Verified $Spec.url $Spec.sha256 $archive

    $tmp = Join-Path $Runtime "_extract-$Name"
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    Write-Host "  展開: $fileName"
    # Windows 10 1803 以降に入っている tar.exe（bsdtar）は .tar.gz と .zip の両方を展開できる
    $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
    $ok = $false
    if (Test-Path -LiteralPath $tar) {
        & $tar -xf $archive -C $tmp
        $ok = ($LASTEXITCODE -eq 0)
    }
    if (-not $ok -and $archive.EndsWith('.zip')) {
        Expand-Archive -LiteralPath $archive -DestinationPath $tmp -Force
        $ok = $true
    }
    if (-not $ok) { Fail "$fileName を展開できませんでした（tar.exe が要ります。Windows 10 1803 以降に入っています）" }

    $inner = Join-Path $tmp $Spec.archive_root
    if (-not (Test-Path -LiteralPath $inner)) { Fail "$fileName の中に $($Spec.archive_root) がありません（sources.json の archive_root を確かめてください）" }
    if (Test-Path -LiteralPath $Target) { Remove-Item -LiteralPath $Target -Recurse -Force }
    Move-Item -LiteralPath $inner -Destination $Target
    Remove-Item -LiteralPath $tmp -Recurse -Force
    Set-Content -LiteralPath $marker -Value $Spec.sha256 -Encoding ASCII
    Write-Host "  $Name $($Spec.version) を入れました"
}

Write-Host ''
Write-Host '=== 0/4 Python と Node.js ==='
New-Item -ItemType Directory -Force -Path $Runtime | Out-Null

# Python は python-build-standalone（astral-sh）の install_only 版。venv・tkinter・ヘッダがあり、sdist のビルドもできる
$pyDir = Join-Path $Runtime 'python'
Install-Archive 'Python' $Sources.python $pyDir
$py = Join-Path $pyDir 'python.exe'
if (-not (Test-Native { & $py -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" })) { Fail "同梱の Python を起動できません: $py" }

# pip は同梱の wheel（ensurepip）から入れる。ネットは要らない
if (-not (Test-Native { & $py -m pip --version })) {
    Write-Host '  pip を入れます（ensurepip）'
    & $py -m ensurepip --default-pip
    if ($LASTEXITCODE -ne 0) { Fail 'pip を入れられませんでした（python -m ensurepip）' }
}

$nodeDir = Join-Path $Runtime 'node'
Install-Archive 'Node.js' $Sources.node $nodeDir
$node = Join-Path $nodeDir 'node.exe'
if (-not (Test-Native { & $node --version })) { Fail "同梱の Node.js を起動できません: $nodeDir\node.exe" }

# 取得したアーカイブは展開したら要らない
Remove-Item -LiteralPath $Downloads -Recurse -Force -ErrorAction SilentlyContinue
exit 0
