# 生成实验报告图表（GDI+，无 Python 依赖）
Add-Type -AssemblyName System.Drawing
$OUT = "C:\path\to\EfficientSAM2\pc\report_assets"
$fontTitle = New-Object System.Drawing.Font("Microsoft YaHei", 22, [System.Drawing.FontStyle]::Bold)
$fontAxis  = New-Object System.Drawing.Font("Microsoft YaHei", 15)
$fontTick  = New-Object System.Drawing.Font("Microsoft YaHei", 13)
$fontNote  = New-Object System.Drawing.Font("Microsoft YaHei", 12)

$cDark  = [System.Drawing.Color]::FromArgb(55, 71, 79)
$cMid   = [System.Drawing.Color]::FromArgb(120, 144, 156)
$cLight = [System.Drawing.Color]::FromArgb(176, 190, 197)
$cLighter = [System.Drawing.Color]::FromArgb(207, 216, 220)
$cGrid  = [System.Drawing.Color]::FromArgb(230, 235, 238)

$brushDark = New-Object System.Drawing.SolidBrush($cDark)
$brushMid = New-Object System.Drawing.SolidBrush($cMid)
$brushLight = New-Object System.Drawing.SolidBrush($cLight)
$brushLighter = New-Object System.Drawing.SolidBrush($cLighter)
$penDark = New-Object System.Drawing.Pen($cDark, 3)
$penDash = New-Object System.Drawing.Pen($cLight, 2)
$penDash.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
$penGrid = New-Object System.Drawing.Pen($cGrid, 1)
$penAxis = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(84, 110, 122), 2)

function New-Canvas($w, $h) {
    $b = New-Object System.Drawing.Bitmap($w, $h)
    $g = [System.Drawing.Graphics]::FromImage($b)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAlias
    $g.Clear([System.Drawing.Color]::White)
    return @($b, $g)
}

$sfCenter = New-Object System.Drawing.StringFormat
$sfCenter.Alignment = [System.Drawing.StringAlignment]::Center

# ================= 图 1：memattn 耗时 vs token 数 =================
$W = 1440; $H = 800
$bc = New-Canvas $W $H; $bmp = $bc[0]; $g = $bc[1]
$g.DrawString("记忆注意力耗时随记忆长度线性增长：token 越少越快", $fontTitle, $brushDark, ($W/2), 24, $sfCenter)
$L = 150; $R = 80; $Tp = 110; $Bt = 120
$pw = $W - $L - $R; $ph = $H - $Tp - $Bt
$xmax = 30.0; $ymax = 8.0
function P1($x, $y) { return @($L + ($x / $xmax) * $pw, $Tp + $ph - ($y / $ymax) * $ph) }
# 网格 + 刻度
for ($yv = 0; $yv -le 8; $yv += 2) {
    $p = P1 0 $yv
    $g.DrawLine($penGrid, $L, $p[1], $W - $R, $p[1])
    $g.DrawString("$yv", $fontTick, $brushMid, ($L - 18), ($p[1] - 11), (New-Object System.Drawing.StringFormat -Property @{ Alignment = "Far" }))
}
foreach ($xv in @(0, 5, 10, 15, 20, 25, 30)) {
    $p = P1 $xv 0
    $g.DrawLine($penGrid, $p[0], $Tp, $p[0], ($Tp + $ph))
    $g.DrawString("$xv", $fontTick, $brushMid, $p[0], ($Tp + $ph + 12), $sfCenter)
}
$g.DrawLine($penAxis, $L, $Tp, $L, ($Tp + $ph))
$g.DrawLine($penAxis, $L, ($Tp + $ph), ($W - $R), ($Tp + $ph))
$g.DrawString("参与注意力的记忆 token 数（千）", $fontAxis, $brushDark, ($L + $pw/2), ($H - 52), $sfCenter)
# 纵轴标题（旋转）
$g.TranslateTransform(44, ($Tp + $ph/2)); $g.RotateTransform(-90)
$g.DrawString("耗时（秒）", $fontAxis, $brushDark, 0, 0, $sfCenter)
$g.ResetTransform()
# 数据线
$toks = @(4.1, 8.2, 12.3, 16.4, 20.5, 24.6, 28.7)
$tms  = @(2.13, 2.97, 4.15, 4.68, 5.50, 6.27, 7.27)
$pts = @()
for ($i = 0; $i -lt $toks.Count; $i++) { $p = P1 $toks[$i] $tms[$i]; $pts += New-Object System.Drawing.PointF($p[0], $p[1]) }
$g.DrawLines($penDark, [System.Drawing.PointF[]]$pts)
foreach ($p in $pts) { $g.FillEllipse($brushDark, ($p.X - 6), ($p.Y - 6), 12, 12) }
# 平台期虚线
$pp = P1 0 7.05
$g.DrawLine($penDash, $L, $pp[1], ($W - $R), $pp[1])
$g.DrawString("平台期 ≈7.05 秒（记忆窗口 7 帧已打满，之后不再变贵）", $fontNote, $brushMid, ($L + 30), ($pp[1] - 30))
# 端点标注
$p0 = $pts[0]; $pN = $pts[6]
$g.DrawString("2.1 秒", $fontTick, $brushDark, $p0.X, ($p0.Y - 34), $sfCenter)
$g.DrawString("7.3 秒", $fontTick, $brushDark, $pN.X, ($pN.Y - 34), $sfCenter)
$bmp.Save("$OUT\chart_memattn_scaling.png", [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()

# ================= 图 2：单帧耗时构成（横向堆叠条） =================
$bc = New-Canvas 1440 620; $bmp = $bc[0]; $g = $bc[1]
$g.DrawString("一帧约 11.1 秒花在哪里：记忆注意力占六成以上", $fontTitle, $brushDark, 720, 24, $sfCenter)
$parts = @(
    @("记忆注意力 memattn", 7.05, $brushDark),
    @("图像编码器 encoder", 3.74, $brushMid),
    @("掩码解码 samheads", 0.13, $brushLight),
    @("记忆编码 memenc", 0.21, $brushLighter)
)
$total = 11.13
$barX = 90; $barY = 200; $barW = 1260; $barH = 150
$x = $barX
$white = [System.Drawing.Brushes]::White
foreach ($p in $parts) {
    $w = [int]($p[1] / $total * $barW)
    $g.FillRectangle($p[2], $x, $barY, $w, $barH)
    if ($p[1] -gt 1) {
        $g.DrawString($p[0], $fontAxis, $white, ($x + $w/2), ($barY + 40), $sfCenter)
        $g.DrawString(("{0:N2} 秒（{1:N0}%）" -f $p[1], ($p[1]/$total*100)), $fontTick, $white, ($x + $w/2), ($barY + 78), $sfCenter)
    }
    $x += $w
}
# 小分段标注放条外
$g.DrawString("掩码解码 0.13s + 记忆编码 0.21s（合计约 3%）", $fontNote, $brushMid, ($x - 320), ($barY + $barH + 24))
$g.DrawString("0", $fontTick, $brushMid, $barX, ($barY + $barH + 60), $sfCenter)
$g.DrawString("11.1 秒", $fontTick, $brushMid, ($barX + $barW), ($barY + $barH + 60), $sfCenter)
$bmp.Save("$OUT\chart_frame_breakdown.png", [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()

# ================= 图 3：18 帧逐帧堆叠柱 =================
$frames = Get-Content "$OUT\frames.json" -Raw | ConvertFrom-Json
$W = 1560; $H = 820
$bc = New-Canvas $W $H; $bmp = $bc[0]; $g = $bc[1]
$g.DrawString("逐帧耗时：记忆随帧数累积，注意力越来越贵", $fontTitle, $brushDark, ($W/2), 24, $sfCenter)
$L = 130; $R = 60; $Tp = 120; $Bt = 130
$pw = $W - $L - $R; $ph = $H - $Tp - $Bt
$ymax = 12.0
for ($yv = 0; $yv -le 12; $yv += 3) {
    $yy = $Tp + $ph - ($yv / $ymax) * $ph
    $g.DrawLine($penGrid, $L, $yy, ($W - $R), $yy)
    $g.DrawString("$yv", $fontTick, $brushMid, ($L - 16), ($yy - 11), (New-Object System.Drawing.StringFormat -Property @{ Alignment = "Far" }))
}
$g.DrawLine($penAxis, $L, $Tp, $L, ($Tp + $ph))
$g.DrawLine($penAxis, $L, ($Tp + $ph), ($W - $R), ($Tp + $ph))
$n = $frames.Count
$slot = $pw / $n
$bw = [int]($slot * 0.68)
for ($i = 0; $i -lt $n; $i++) {
    $f = $frames[$i]
    $x = [int]($L + $i * $slot + ($slot - $bw) / 2)
    $vals = @(@($f.enc/1000, $brushMid), @($f.memattn/1000, $brushDark), @($f.sam/1000, $brushLight), @($f.memenc/1000, $brushLighter))
    $yBase = $Tp + $ph
    foreach ($v in $vals) {
        $hh = [int]($v[0] / $ymax * $ph)
        if ($hh -gt 0) { $g.FillRectangle($v[1], $x, ($yBase - $hh), $bw, $hh) }
        $yBase -= $hh
    }
    if ($i % 2 -eq 0) { $g.DrawString("$i", $fontTick, $brushMid, ($x + $bw/2), ($Tp + $ph + 12), $sfCenter) }
}
$g.DrawString("帧序号（第 0 帧为标注帧，不做记忆注意力）", $fontAxis, $brushDark, ($L + $pw/2), ($H - 52), $sfCenter)
$g.TranslateTransform(40, ($Tp + $ph/2)); $g.RotateTransform(-90)
$g.DrawString("耗时（秒）", $fontAxis, $brushDark, 0, 0, $sfCenter)
$g.ResetTransform()
# 图例
$lg = @(@("图像编码器", $brushMid), @("记忆注意力", $brushDark), @("掩码解码", $brushLight), @("记忆编码", $brushLighter))
$lx = $W - 560
foreach ($e in $lg) {
    $g.FillRectangle($e[1], $lx, 96, 26, 16)
    $g.DrawString($e[0], $fontTick, $brushDark, ($lx + 34), 92)
    $lx += 140
}
$bmp.Save("$OUT\chart_frames.png", [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()

# ================= 图 4：PC vs Mate 80 对比 =================
$W = 1440; $H = 760
$bc = New-Canvas $W $H; $bmp = $bc[0]; $g = $bc[1]
$g.DrawString("同一模型：手机 CPU 耗时约为 PC 的 3~5 倍", $fontTitle, $brushDark, ($W/2), 24, $sfCenter)
$names = @("图像编码器", "记忆注意力（稳态）", "记忆编码", "掩码解码")
$x86 = @(1.058, 2.241, 0.044, 0.024)
$m80 = @(3.743, 7.05, 0.21, 0.134)
$L = 130; $R = 60; $Tp = 120; $Bt = 120
$pw = $W - $L - $R; $ph = $H - $Tp - $Bt
$ymax = 8.0
for ($yv = 0; $yv -le 8; $yv += 2) {
    $yy = $Tp + $ph - ($yv / $ymax) * $ph
    $g.DrawLine($penGrid, $L, $yy, ($W - $R), $yy)
    $g.DrawString("$yv", $fontTick, $brushMid, ($L - 16), ($yy - 11), (New-Object System.Drawing.StringFormat -Property @{ Alignment = "Far" }))
}
$g.DrawLine($penAxis, $L, $Tp, $L, ($Tp + $ph))
$g.DrawLine($penAxis, $L, ($Tp + $ph), ($W - $R), ($Tp + $ph))
$slot = $pw / 4
for ($i = 0; $i -lt 4; $i++) {
    $cx = $L + $i * $slot + $slot / 2
    $bw = 110
    foreach ($pair in @(@($x86[$i], $brushLight, -1), @($m80[$i], $brushDark, 1))) {
        $hh = [Math]::Max(2, [int]($pair[0] / $ymax * $ph))
        $bx = [int]($cx + $pair[2] * 62 - $bw/2)
        $g.FillRectangle($pair[1], $bx, ($Tp + $ph - $hh), $bw, $hh)
        $g.DrawString(("{0:N2}" -f $pair[0]), $fontTick, $brushDark, ($bx + $bw/2), ($Tp + $ph - $hh - 28), $sfCenter)
    }
    $g.DrawString($names[$i], $fontTick, $brushDark, $cx, ($Tp + $ph + 14), $sfCenter)
}
$g.TranslateTransform(40, ($Tp + $ph/2)); $g.RotateTransform(-90)
$g.DrawString("耗时（秒）", $fontAxis, $brushDark, 0, 0, $sfCenter)
$g.ResetTransform()
$g.FillRectangle($brushLight, ($W - 480), 96, 26, 16); $g.DrawString("PC（x86 CPU）", $fontTick, $brushDark, ($W - 446), 92)
$g.FillRectangle($brushDark, ($W - 270), 96, 26, 16); $g.DrawString("Mate 80（麒麟 9020）", $fontTick, $brushDark, ($W - 236), 92)
$bmp.Save("$OUT\chart_x86_vs_m80.png", [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()

Write-Output "charts done"
