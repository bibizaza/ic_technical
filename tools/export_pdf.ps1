# PowerPoint deck -> PDF with PowerPoint itself (intern PC, tools/market_publish.py pdf).
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File export_pdf.ps1 <in.pptx> <out.pdf>   (absolute paths)
# Opens the deck read-only without a window and closes only that deck; PowerPoint is quit only if this script
# started it, so a presentation someone has open is never touched.
$ErrorActionPreference = 'Stop'
$in, $out = $args[0], $args[1]
$wasRunning = [bool](Get-Process POWERPNT -ErrorAction SilentlyContinue)
$app = New-Object -ComObject PowerPoint.Application
try {
    $deck = $app.Presentations.Open($in, -1, 0, 0)   # ReadOnly = msoTrue, Untitled = msoFalse, WithWindow = msoFalse
    try { $deck.SaveAs($out, 32) } finally { $deck.Close() }   # 32 = ppSaveAsPDF
} finally {
    if (-not $wasRunning -and $app.Presentations.Count -eq 0) { $app.Quit() }
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app)
}
