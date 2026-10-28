param([ValidateSet('file', 'folder')][string]$Kind)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$ownerWindow = New-Object System.Windows.Forms.Form
$ownerWindow.TopMost = $true
$ownerWindow.ShowInTaskbar = $false
$ownerWindow.Opacity = 0
$ownerWindow.Width = 1
$ownerWindow.Height = 1
if ($Kind -eq 'file') {
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = '选择照片（可按 Ctrl 或 Shift 多选）'
    $dialog.Filter = '照片和RAW文件|*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.tif;*.tiff;*.gif;*.avif;*.heic;*.heif;*.3fr;*.arw;*.bay;*.cr2;*.cr3;*.crw;*.dcr;*.dcs;*.dng;*.erf;*.fff;*.gpr;*.iiq;*.k25;*.kdc;*.mdc;*.mef;*.mos;*.mrw;*.nef;*.nrw;*.orf;*.pef;*.raf;*.raw;*.rw2;*.rwl;*.sr2;*.srf;*.srw;*.x3f|所有文件|*.*'
    $dialog.Multiselect = $true
    $dialog.CheckFileExists = $true
} else {
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = '选择照片文件夹'
    $dialog.ShowNewFolderButton = $false
}
try {
    $ownerWindow.Show()
    $result = $dialog.ShowDialog($ownerWindow)
    $selectedPaths = @()
    if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
        if ($Kind -eq 'file') { $selectedPaths = @($dialog.FileNames) }
        else { $selectedPaths = @($dialog.SelectedPath) }
    }
    [pscustomobject]@{ paths = $selectedPaths; cancelled = ($selectedPaths.Count -eq 0) } | ConvertTo-Json -Compress
} finally {
    $dialog.Dispose()
    $ownerWindow.Dispose()
}
