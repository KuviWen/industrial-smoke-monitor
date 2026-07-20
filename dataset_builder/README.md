# Industrial Smoke Dataset Builder

這是母專案 `industrial-smoke-monitor` 底下的資料製作子專案。它把現場影片轉成可供母專案 YOLO11 instance segmentation 直接使用的資料集，並提供一個不需要 Streamlit 的 Tkinter GUI 進行人工標註。

文件可離線開啟：

- [YOLO11 特色與工作原理導覽](../intro_YOLO11/)
- [專案架構介紹](docs/architecture.html)
- [操作說明書](docs/operations.html)

若不熟悉 Detect 與 Segment 的差異，建議先閱讀上面的 YOLO11 導覽，再理解本工具為什麼輸出 polygon label，以及這些資料如何交給母專案訓練。

## 產出格式

GUI 儲存後會建立以下結構：

```text
data/processed/video_yolo/
├── dataset.yaml
├── manifest.csv
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

`dataset.yaml` 可直接交給母專案的 `scripts/train.py` 或 `scripts/validate.py`。每一行 label 是 YOLO11 segmentation polygon；沒有煙的影格則是空白 `.txt`，但只有在人工確認後才能儲存成負樣本。

## 左右翻轉資料增強

GUI 的「是否同時生成左右翻轉的副本」選項只建議在 `split=train` 時勾選。儲存一張已人工確認的影像時，工具會同時產生：

```text
images/train/camera_f00000001.jpg
labels/train/camera_f00000001.txt
images/train/camera_f00000001_flip.jpg
labels/train/camera_f00000001_flip.txt
```

翻轉副本使用 OpenCV 左右翻轉影像，並將每個 polygon 的 x 座標依 `x' = width - 1 - x` 轉換，y 座標與 polygon 順序保留不變，因此不需要重新手動畫遮罩。原始影像與副本都會寫入 `manifest.csv`；副本的 `label_status` 是 `reviewed_augmented_horizontal_flip`。沒有煙的空白 label 也能正確產生翻轉副本。

不要對 `val` 或 `test` 開啟這個選項，避免驗證資料被擴增而使評估失真；GUI 在非 `train` split 會提示先切換或取消。

## 快速開始（Windows / Anaconda）

從母專案根目錄執行：

```powershell
conda activate YOLO11
python dataset_builder/scripts/extract_frames.py `
  --video data/raw/videos/chimney_01.mp4 `
  --output dataset_builder/data/staging `
  --every-seconds 1

python dataset_builder/gui/annotator_gui.py `
  --video data/raw/videos/chimney_01.mp4 `
  --output data/processed/video_yolo `
  --weights models/best.pt `
  --split train

python dataset_builder/scripts/check_dataset.py `
  --dataset data/processed/video_yolo
```

`--weights` 是可選的本機 YOLO11-seg 權重；離線環境不會自動下載模型。沒有權重時仍可完全手動標註。

GUI 與抽幀工具接受 `.mp4`、`.avi`、`.mov`、`.mkv`、`.ts` 與 `.3gp`。`.3gp` 能否讀取取決於目前 OpenCV/FFmpeg 後端是否包含 3GPP codec。

## GitHub 注意事項

影片、生成的資料集、模型權重與 runtime 檔案都不應提交到 Git。請只提交此子專案的 Python 原始碼、HTML 文件、測試與設定；大型資料與權重使用內部檔案伺服器、Git LFS 或其他受控儲存。
