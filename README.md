# Industrial Smoke Monitor

以 YOLO11 instance segmentation 判斷煙囪是否正在冒煙，透過 RTSP 讀取現場攝影機影像，並以連續影格判斷降低誤報，最後經由可達的內部 SMTP relay 寄出 Email。此專案適合先在可連外的訓練電腦建立模型，再把模型、Python 環境與程式複製到隔離外網的現場電腦。

本專案也附有一份以實際 IJmond 訓練紀錄為案例的評估教材，示範如何從 P／R、mAP、曲線、預測圖與現場事件指標判斷模型是否值得繼續調整，並依煙囪在畫面中很小等情境規劃資料與參數實驗。

若第一次接觸 YOLO11，建議先閱讀 `intro_YOLO11/` 的技術導覽，了解 Detect、Segment、Backbone、Neck、Head、遷移學習、mAP、信心門檻與部署格式，再閱讀本專案的訓練與部署文件。

## 先看這些文件

- [專案架構介紹](docs/architecture.html)
- [YOLO11 特色與工作原理導覽](intro_YOLO11/)
- [手把手準備離線 Windows Python 3.11.15 環境](docs/offline_win_amd64_python311_cpu_install_guide.html)
- [從訓練到部署操作書](docs/operations.html)
- [訓練結果評估與調參指南（本次 IJmond 案例）](supplements/training_result_evaluation_guide.html)
- [影片資料製作](dataset_builder)
- [影片切割](video_splitter)
- [報警資料檢查／重標註 GUI](alert_reviewer)

直接以瀏覽器開啟即可，不需要網路或額外 Web server。

## 快速開始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux:   source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

# 把 IJmond zip 解壓到 data/raw/ijmond/ 後：
python scripts/prepare_ijmond.py \
  --input data/raw/ijmond \
  --output data/processed/ijmond_yolo \
  --split-strategy camera \
  --overwrite

python scripts/train.py \
  --data configs/ijmond.yaml \
  --model yolo11n-seg.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --name smoke_yolo11n

python scripts/validate.py \
  --weights runs/train/smoke_yolo11n/weights/best.pt \
  --data configs/ijmond.yaml \
  --split test \
  --device 0
```

`requirements.txt` 預設啟用 CUDA 12.4 的 PyTorch，適合 GPU 訓練；沒有 GPU 的訓練或現場電腦，請先依 [完整操作書](docs/operations.html) 將 CUDA torch 行註解並啟用 CPU torch 行，再安裝套件。NumPy 固定為 `1.26`，OpenCV 使用標準 `opencv-python`。

將最佳權重複製成 `models/best.pt`，再將非隱藏設定範例複製為 `configs/monitor_settings.env`，填入 RTSP、ROI、SMTP relay 與收件人：

```powershell
Copy-Item configs\monitor_settings.env.example configs\monitor_settings.env
notepad configs\monitor_settings.env
```

```bash
python scripts/check_rtsp.py --settings configs/monitor_settings.env --seconds 15
python scripts/test_email.py --settings configs/monitor_settings.env
python scripts/run_monitor.py --settings configs/monitor_settings.env
```

若要先觀察模型而不寄信，設定 `SHADOW_MODE=true`；若要以瀏覽器觀看即時標註影像，設定 `ALLOW_LIVE_STREAMING=true`，然後開啟 `http://127.0.0.1:8765/`。煙霧告警預設會保存到 `data/runtime/alerts/`，包含 JPG 與同名 JSON，供 [報警資料檢查／重標註 GUI](alert_reviewer/docs/operations.html) 人工確認。

正式設定檔固定使用 `configs/monitor_settings.env`；專案不再讀取或建立根目錄的隱藏設定檔。現場 Windows 電腦需要安裝 Python 3.11.15 與相依套件；隔離環境請依離線安裝指南準備 Windows wheel／Conda 套件，再以 `deploy/install_windows.ps1` 安裝。

## 目錄說明

```text
configs/       monitor_settings.env.example 與 YOLO 資料集設定
data/raw/      原始資料集；不提交 Git
data/processed/轉換後的 YOLO 資料集；不提交 Git
data/runtime/  現場 log、JSONL 紀錄與告警證據圖
models/        放置 best.pt；大型權重不提交 Git
scripts/       資料轉換、訓練、驗證、匯出與啟動入口
src/           RTSP、YOLO、時間判斷、Email、儲存模組
deploy/        Linux systemd 與 Windows 啟動腳本
docs/          可離線閱讀的 HTML 文件
intro_YOLO11/  YOLO11 特色、架構、訓練、評估與部署入門
tests/         告警狀態與設定測試
dataset_builder/ 影片抽幀、YOLO11-seg 標註、資料檢查與 GUI
video_splitter/  影片預覽、單段/批次切割與輸出
alert_reviewer/ 報警影像檢查、人工重標註與回送訓練資料集
```

## 最重要的現場限制

攝影機電腦若完全不能連到 SMTP relay，就不可能直接寄出 Email。Email 需要一條到郵件伺服器的網路路徑；因此現場需由 IT 提供「只允許到內部 SMTP relay」的網路規則，或部署內部郵件佇列／轉送服務。不要把公開信箱密碼寫進 Git，也不要假設隔離網路可以直接連 Gmail、Outlook 或其他外部 SMTP。

## 資料與授權

IJmond 資料集的原始標註不是 YOLO 格式。本專案的 `prepare_ijmond.py` 會讀取資料集的 cropped images、raster masks 與官方 camera/timestamp split，將低／高不透明度合併成一個 `smoke` 類別，再產生 YOLO polygon labels。資料集使用 CC BY 4.0，請保留作者與資料集引用；Ultralytics YOLO11 的授權條件也必須在部署前確認。詳細說明見 `THIRD_PARTY_NOTICES.md`。
