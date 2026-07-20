# Industrial Video Splitter

這是 `industrial-smoke-monitor` 底下的影片切割子專案。它使用 Python 內建 Tkinter、OpenCV 與 Pillow，讓操作人員在 GUI 中預覽影片、選擇起訖範圍，或依固定秒數批次切成多個影片檔。

文件可離線開啟：

- [YOLO11 特色與工作原理導覽](../intro_YOLO11/)
- [專案架構介紹](docs/architecture.html)
- [操作說明書](docs/operations.html)

## 快速開始

從母專案根目錄的 Anaconda Prompt 執行：

```powershell
conda activate YOLO11
python video_splitter/gui/video_splitter_gui.py `
  --video data/raw/videos/chimney_01.mp4 `
  --output video_splitter/data/output
```

GUI 支援：

- 影格預覽、時間軸拖曳、上一/下一影格與影格索引跳轉
- 單段切割：指定開始與結束時間，輸出一個片段
- 批次切割：指定每段長度與起點間隔，可產生不重疊或重疊片段
- 輸出檔名前綴、輸出資料夾、同名檔案覆寫選項
- 背景執行、進度列與取消操作

可選取的輸入副檔名包含 `.mp4`、`.avi`、`.mov`、`.mkv`、`.ts` 與 `.3gp`。`.3gp` 是否能實際解碼，仍取決於目前 OpenCV/FFmpeg 後端是否包含對應的 3GPP codec。

## 輸出說明

輸出檔預設為：

```text
video_splitter/data/output/<prefix>_0001.mp4
video_splitter/data/output/<prefix>_0002.mp4
```

OpenCV 會重新編碼影片影像，因此輸出通常不包含原始音訊軌。這個子專案的目標是製作用於煙霧影像資料製作的影片片段；若必須保留音訊，應使用另外的 FFmpeg 流程，不要把未驗證的音訊合併流程混入本工具。

輸出片段可直接交給母專案的 `dataset_builder/` 做抽幀與標註。

## GitHub 注意事項

原始影片與切割後影片都不應提交到 Git；子專案的 `.gitignore` 已排除影片檔與 `data/output`，GitHub 只保存程式、文件、測試與空資料夾佔位檔。
