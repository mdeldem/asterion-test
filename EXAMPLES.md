Convertir plusieurs images :
```powershell
uv run asterion-test `
"data/IC5070_V1490_Cyg_2026_06_20_stack_10x180sec-SI.fits" `
"data/IC5070_V1490_Cyg_2026_06_20_stack_10x180sec-SR.fits" `
"data/IC5070_V1490_Cyg_2026_06_20_stack_10x180sec-V.fits" `
--save-fits --save-jpg --output-dir "data/output"
```

Convertir une série d'images :
```powershell
uv run asterion-test `
--series "E:/CCD/2026/20260627_C11/NSV 11142/fits/NSV 11142-001.fits" `
 --save-fits --save-jpg --output-dir "output/20260627_NSV_11142"
```