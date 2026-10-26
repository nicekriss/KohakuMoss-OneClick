# Puts a "KohakuMoss 원클릭 LoRA" shortcut with the grasshopper icon on the desktop, pointing at this folder's launcher.
$desk = [Environment]::GetFolderPath('Desktop')
$lnk = Join-Path $desk 'KohakuMoss 원클릭 LoRA.lnk'
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
$s.TargetPath = Join-Path $PSScriptRoot 'KohakuMoss_원클릭학습.bat'
$s.WorkingDirectory = $PSScriptRoot
$s.IconLocation = (Join-Path $PSScriptRoot 'icon.ico') + ',0'
$s.Description = 'MOSS-Image LoRA 원클릭 학습기'
$s.Save()
Write-Host ''
Write-Host " 바탕화면에 바로가기를 만들었어요: $lnk"
Write-Host ''
