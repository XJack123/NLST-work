# 根目录与输出目录（使用纯ASCII路径，避免中文路径问题）
$ROOT = "D:\data\manifest-1747923321115\NLST"
$OUT  = "C:\Users\15031\Desktop\NLSTNIfTIdata"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

# 随机抽10个病例
$patients = (Get-ChildItem -Directory $ROOT) | Get-Random -Count ([math]::Min(10, (Get-ChildItem -Directory $ROOT).Count))

foreach ($p in $patients) {
  foreach ($study in Get-ChildItem -Directory $p.FullName) {
    foreach ($se in Get-ChildItem -Directory $study.FullName) {
      # 相对路径与输出目录
      $rel  = $se.FullName.Substring($ROOT.Length).TrimStart('\\')
      $dest = Join-Path $OUT $rel
      New-Item -ItemType Directory -Force -Path $dest | Out-Null

      # 用相对路径生成唯一且合法的文件名（ASCII）
      $safe = ($rel -replace '[\\/:*?"<>| ]','_')

      # 直接输出到ASCII路径
      dcm2niix -z y -f "$safe" -o "$dest" "$( $se.FullName )"
    }
  }
}
