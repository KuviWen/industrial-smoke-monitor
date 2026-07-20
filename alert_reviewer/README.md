# 報警資料檢查與重標註子專案

`alert_reviewer/` 是母專案的人工複核工具。監控程序會把偵測到的 smoke 影像與 JSON 評分寫到 `data/runtime/alerts/`；使用者可在 GUI 中查看 ROI、box、mask、conf、area、cls 與原始 polygon，修正為 `smoke` 或 `no_smoke`，再選擇：

- 保留在原地：更新 alert JSON，並在旁邊寫出 YOLO label；
- 移動到訓練資料集：輸出到母專案可直接使用的 `images/{split}`、`labels/{split}`、`dataset.yaml` 與 `manifest.csv`，再移除原告警檔；
- 刪除：刪除告警影像、JSON 與 label，不會刪除其他 runtime log。

告警資料夾與 `data/processed/` 分開，未經人工確認的樣本不會自動進入訓練資料集。

## 文件

- [子專案架構](docs/architecture.html)
- [GUI 操作說明書](docs/operations.html)

## 安裝與啟動

在母專案根目錄的 Python 3.11.15 環境執行：

```powershell
python -m pip install -r alert_reviewer/requirements.txt
python alert_reviewer/gui/alert_reviewer_gui.py `
  --input data/runtime/alerts `
  --dataset data/processed/field_yolo `
  --split train
```

Windows 官方 Python 安裝通常已包含 Tkinter；若 GUI 啟動時出現 `No module named tkinter`，請重新安裝 Python 並勾選標準庫／Tcl-Tk 元件。這個工具不需要 `best.pt`、Ultralytics 或 GPU。

## 建議流程

1. 先以 `SHADOW_MODE=true` 讓母專案監測但不寄信，累積真實現場資料。
2. 用 GUI 先處理明顯誤報，再把確認過的 smoke 與 no_smoke 樣本移到 `data/processed/field_yolo`。
3. 執行母專案的資料檢查，確認 train/val/test 分布與標籤品質。
4. 用新資料建立新模型版本，保留原本的 `best.pt` 以便回復。
