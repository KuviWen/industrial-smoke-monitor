"""Tkinter frame annotation GUI for the smoke YOLO11-seg dataset.

The GUI intentionally uses only Tkinter, OpenCV, Pillow, NumPy, and the
already-installed Ultralytics package.  A local ``--weights`` file is
optional: manual polygon annotation works without loading a model, while a
YOLO11-seg checkpoint can provide a starting mask that the reviewer can
replace or keep.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

CHILD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHILD_ROOT / "src"))

from smoke_dataset_builder.dataset import YoloDatasetWriter  # noqa: E402
from smoke_dataset_builder.video import (  # noqa: E402
    SUPPORTED_VIDEO_EXTENSIONS,
    read_video_info,
    safe_video_stem,
)
from smoke_dataset_builder.yolo import result_to_polygons  # noqa: E402
from smoke_dataset_builder.roi import (  # noqa: E402
    Roi,
    crop_frame,
    format_roi,
    parse_roi,
    validate_roi,
)


class AnnotatorApp(tk.Tk):
    """One-video-at-a-time polygon annotator."""

    CANVAS_WIDTH = 960
    CANVAS_HEIGHT = 600

    def __init__(
        self,
        video: str | None = None,
        output: str = "data/processed/video_yolo",
        weights: str | None = None,
        split: str = "train",
        roi: str | None = None,
    ) -> None:
        super().__init__()
        self.title("Industrial Smoke Dataset Builder — YOLO11-seg")
        self.geometry("1160x820")
        self.minsize(900, 680)

        self.video_path: Path | None = None
        self.capture: cv2.VideoCapture | None = None
        self.info = None
        self.frame = None
        self.frame_index = 0
        self.photo: ImageTk.PhotoImage | None = None
        self.display_scale = 1.0
        self.display_offset = (0.0, 0.0)
        self.display_size = (0, 0)
        self.polygons: list[list[tuple[float, float]]] = []
        self.current_polygon: list[tuple[float, float]] = []
        self.model = None
        self.roi: Roi | None = None

        self.video_var = tk.StringVar(value=video or "")
        self.output_var = tk.StringVar(value=output)
        self.weights_var = tk.StringVar(value=weights or "")
        self.split_var = tk.StringVar(value=split if split in {"train", "val", "test"} else "train")
        self.roi_var = tk.StringVar(value=roi or "")
        self.use_roi_var = tk.BooleanVar(value=bool(roi and roi.strip()))
        self.horizontal_flip_var = tk.BooleanVar(value=False)
        self.frame_var = tk.StringVar(value="Frame: —")
        self.status_var = tk.StringVar(
            value="選擇影片後，點擊畫面建立煙霧多邊形；沒有煙也要人工確認後按儲存。"
        )

        self._build_widgets()
        self.protocol("WM_DELETE_WINDOW", self._close)
        if video:
            self.after(100, lambda: self._open_video(Path(video)))

    def _build_widgets(self) -> None:
        settings = ttk.Frame(self, padding=8)
        settings.pack(fill="x")
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="影片").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(settings, textvariable=self.video_var).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Button(settings, text="選擇影片", command=self._choose_video).grid(row=0, column=2, padx=4)
        ttk.Button(settings, text="開啟", command=self._open_video_from_entry).grid(row=0, column=3)

        ttk.Label(settings, text="輸出資料集").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(settings, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Button(settings, text="選擇資料夾", command=self._choose_output).grid(row=1, column=2, padx=4)
        ttk.Label(settings, text="split").grid(row=1, column=3, sticky="e", padx=(10, 4))
        ttk.Combobox(
            settings,
            textvariable=self.split_var,
            values=("train", "val", "test"),
            width=8,
            state="readonly",
        ).grid(row=1, column=4, sticky="e")

        ttk.Label(settings, text="YOLO11 權重（可選）").grid(
            row=2, column=0, sticky="w", padx=(0, 6), pady=3
        )
        ttk.Entry(settings, textvariable=self.weights_var).grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Button(settings, text="選擇權重", command=self._choose_weights).grid(row=2, column=2, padx=4)
        ttk.Button(settings, text="載入/自動預標註", command=self._predict).grid(
            row=2, column=3, columnspan=2, sticky="ew", padx=(4, 0)
        )

        ttk.Label(settings, text="ROI x1,y1,x2,y2").grid(
            row=3, column=0, sticky="w", padx=(0, 6), pady=3
        )
        ttk.Entry(settings, textvariable=self.roi_var).grid(
            row=3, column=1, sticky="ew", pady=3
        )
        ttk.Checkbutton(
            settings,
            text="使用 ROI 裁切後標註",
            variable=self.use_roi_var,
        ).grid(row=3, column=2, sticky="w", padx=4)
        ttk.Button(settings, text="套用 ROI", command=self._apply_roi).grid(
            row=3, column=3, columnspan=2, sticky="ew", padx=(4, 0)
        )

        controls = ttk.Frame(self, padding=(8, 0, 8, 8))
        controls.pack(fill="x")
        ttk.Button(controls, text="上一影格", command=lambda: self._move_frame(-1)).pack(side="left", padx=(0, 4))
        ttk.Button(controls, text="下一影格", command=lambda: self._move_frame(1)).pack(side="left", padx=4)
        ttk.Label(controls, text="影格索引").pack(side="left", padx=(18, 4))
        self.frame_entry = ttk.Entry(controls, width=10)
        self.frame_entry.pack(side="left")
        self.frame_entry.bind("<Return>", lambda _event: self._jump_frame())
        ttk.Button(controls, text="跳到", command=self._jump_frame).pack(side="left", padx=4)
        ttk.Button(controls, text="完成多邊形", command=self._finish_polygon).pack(side="left", padx=(18, 4))
        ttk.Button(controls, text="撤銷點/多邊形", command=self._undo).pack(side="left", padx=4)
        ttk.Button(controls, text="清除標註", command=self._clear_annotations).pack(side="left", padx=4)
        ttk.Checkbutton(
            controls,
            text="是否同時生成左右翻轉的副本",
            variable=self.horizontal_flip_var,
        ).pack(side="left", padx=(18, 4))
        ttk.Button(controls, text="儲存此影格", command=self._save).pack(side="right")

        canvas_frame = ttk.Frame(self, padding=(8, 0, 8, 0))
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.CANVAS_WIDTH,
            height=self.CANVAS_HEIGHT,
            background="#161a1d",
            highlightthickness=1,
            highlightbackground="#59636e",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._canvas_click)
        self.canvas.bind("<Configure>", lambda _event: self._render())

        footer = ttk.Frame(self, padding=8)
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.frame_var).pack(side="left")
        ttk.Label(footer, textvariable=self.status_var, wraplength=850).pack(side="right")

    def _choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇影片",
            filetypes=[
                ("Video", " ".join(f"*{extension}" for extension in SUPPORTED_VIDEO_EXTENSIONS)),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.video_var.set(path)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="選擇 YOLO 資料集輸出資料夾")
        if path:
            self.output_var.set(path)

    def _choose_weights(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇 YOLO11-seg 權重",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.weights_var.set(path)

    def _open_video_from_entry(self) -> None:
        value = self.video_var.get().strip()
        if value:
            self._open_video(Path(value))

    def _open_video(self, path: Path) -> None:
        path = path.expanduser().resolve()
        if not path.is_file():
            messagebox.showerror("找不到影片", str(path))
            return
        try:
            info = read_video_info(path)
            configured_roi = self._configured_roi(info.width, info.height)
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError("OpenCV 無法開啟影片")
        except Exception as exc:  # noqa: BLE001 - UI should show the actual cause
            messagebox.showerror("開啟影片失敗", str(exc))
            return
        if self.capture is not None:
            self.capture.release()
        self.video_path = path
        self.video_var.set(str(path))
        self.info = info
        self.capture = capture
        self.roi = configured_roi
        self.frame_index = 0
        self.model = None
        self._clear_annotations()
        self._read_current_frame()
        self._set_status(
            f"已開啟 {path.name}；影片 {info.width}x{info.height}, "
            f"{info.fps:.2f} FPS, {info.frame_count} frames。"
            + (f"目前使用 ROI {format_roi(self.roi)}，標註與輸出均為裁切後座標。" if self.roi else "目前使用完整影格。")
        )

    def _configured_roi(self, width: int, height: int) -> Roi | None:
        """Read and validate the GUI ROI against the original video size."""

        if not self.use_roi_var.get():
            return None
        roi = parse_roi(self.roi_var.get())
        if roi is None:
            raise ValueError("已勾選使用 ROI，請輸入 x1,y1,x2,y2")
        return validate_roi(roi, width, height)

    def _apply_roi(self) -> None:
        """Apply a new ROI and reload the current original video frame."""

        if self.info is None:
            self._set_status("請先開啟影片，再套用 ROI。")
            return
        if not self._ask_unsaved():
            return
        try:
            roi = self._configured_roi(self.info.width, self.info.height)
        except ValueError as exc:
            messagebox.showerror("ROI 設定錯誤", str(exc))
            return
        self.roi = roi
        self.polygons = []
        self.current_polygon = []
        if self._read_current_frame():
            if roi:
                self._set_status(
                    f"已套用 ROI {format_roi(roi)}；目前顯示裁切後影像，請重新確認標註。"
                )
            else:
                self._set_status("已關閉 ROI；目前顯示完整影格，請重新確認標註。")

    def _read_current_frame(self) -> bool:
        if self.capture is None or self.info is None:
            return False
        self.frame_index = max(0, min(self.frame_index, max(0, self.info.frame_count - 1)))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        ok, frame = self.capture.read()
        if not ok:
            self._set_status(f"無法讀取影格 {self.frame_index}")
            return False
        self.frame = crop_frame(frame, self.roi) if self.roi else frame
        self.frame_var.set(
            f"Frame: {self.frame_index} / {max(0, self.info.frame_count - 1)} "
            f"({self.frame_index / self.info.fps:.2f}s)"
        )
        self._render()
        return True

    def _ask_unsaved(self) -> bool:
        if not self.polygons and not self.current_polygon:
            return True
        answer = messagebox.askyesnocancel(
            "標註尚未儲存",
            "目前影格有尚未儲存的標註。要先儲存嗎？\n\n是：儲存後繼續\n否：放棄標註\n取消：留在目前影格",
        )
        if answer is None:
            return False
        if answer:
            return self._save(show_message=False)
        return True

    def _move_frame(self, delta: int) -> None:
        if self.info is None:
            return
        if not self._ask_unsaved():
            return
        self.frame_index += delta
        self.polygons = []
        self.current_polygon = []
        self._read_current_frame()

    def _jump_frame(self) -> None:
        if self.info is None:
            return
        try:
            target = int(self.frame_entry.get())
        except ValueError:
            self._set_status("影格索引必須是整數")
            return
        if not self._ask_unsaved():
            return
        self.frame_index = target
        self.polygons = []
        self.current_polygon = []
        self._read_current_frame()

    def _canvas_to_image(self, x: float, y: float) -> tuple[float, float] | None:
        if self.frame is None:
            return None
        offset_x, offset_y = self.display_offset
        display_width, display_height = self.display_size
        if not (offset_x <= x <= offset_x + display_width and offset_y <= y <= offset_y + display_height):
            return None
        image_x = (x - offset_x) / self.display_scale
        image_y = (y - offset_y) / self.display_scale
        height, width = self.frame.shape[:2]
        return (
            min(max(image_x, 0.0), width - 1.0),
            min(max(image_y, 0.0), height - 1.0),
        )

    def _to_canvas(self, point: tuple[float, float]) -> tuple[float, float]:
        offset_x, offset_y = self.display_offset
        return (
            offset_x + point[0] * self.display_scale,
            offset_y + point[1] * self.display_scale,
        )

    def _canvas_click(self, event) -> None:
        point = self._canvas_to_image(event.x, event.y)
        if point is None:
            return
        self.current_polygon.append(point)
        self._set_status(f"目前多邊形已有 {len(self.current_polygon)} 個點；完成後按「完成多邊形」。")
        self._render()

    def _finish_polygon(self) -> None:
        if len(self.current_polygon) < 3:
            self._set_status("多邊形至少需要 3 個點。")
            return
        self.polygons.append(self.current_polygon[:])
        self.current_polygon = []
        self._set_status(f"已完成第 {len(self.polygons)} 個煙霧多邊形。")
        self._render()

    def _undo(self) -> None:
        if self.current_polygon:
            self.current_polygon.pop()
        elif self.polygons:
            self.polygons.pop()
        self._set_status("已撤銷最後一個點或多邊形。")
        self._render()

    def _clear_annotations(self) -> None:
        self.polygons = []
        self.current_polygon = []
        self._render()

    def _load_model(self):
        weights_text = self.weights_var.get().strip()
        if not weights_text:
            raise ValueError("請先指定本機 YOLO11-seg .pt 權重；離線環境不會自動下載。")
        weights = Path(weights_text).expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"權重不存在: {weights}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("目前環境沒有 ultralytics，請依 requirements.txt 安裝。") from exc
        self._set_status("正在載入 YOLO11 權重，第一次可能需要一些時間……")
        self.update_idletasks()
        self.model = YOLO(str(weights))
        return self.model

    def _predict(self) -> None:
        if self.frame is None:
            messagebox.showwarning("尚未開啟影片", "請先開啟影片。")
            return
        try:
            model = self.model or self._load_model()
            result = model.predict(
                source=self.frame,
                imgsz=640,
                conf=0.25,
                device="cpu",
                verbose=False,
            )[0]
            self.polygons = result_to_polygons(result, self.frame.shape, class_id=0)
            self.current_polygon = []
            self._set_status(
                f"自動預標註完成：找到 {len(self.polygons)} 個 smoke mask；請人工修正後再儲存。"
            )
            self._render()
        except Exception as exc:  # noqa: BLE001 - show model/runtime errors in UI
            messagebox.showerror("自動預標註失敗", str(exc))

    def _save(self, show_message: bool = True) -> bool:
        if self.frame is None or self.video_path is None:
            if show_message:
                messagebox.showwarning("尚未準備好", "請先開啟影片並載入影格。")
            return False
        generate_horizontal_flip = self.horizontal_flip_var.get()
        if generate_horizontal_flip and self.split_var.get() != "train":
            messagebox.showwarning(
                "資料增強建議只使用 train",
                "左右翻轉副本應加入 train split；val/test 不建議做資料增強。\n"
                "請切換 split=train，或取消此選項後再儲存。",
            )
            return False
        if self.current_polygon:
            answer = messagebox.askyesno(
                "多邊形尚未完成",
                "目前還有未按「完成多邊形」的點。要捨棄這些點並繼續儲存嗎？",
            )
            if not answer:
                return False
            self.current_polygon = []
        try:
            writer = YoloDatasetWriter(self.output_var.get().strip() or "data/processed/video_yolo")
            stem = f"{safe_video_stem(self.video_path)}_f{self.frame_index:08d}"
            image_path, label_path = writer.save_sample(
                self.frame,
                self.polygons,
                self.split_var.get(),
                stem,
                source_video=self.video_path,
                frame_index=self.frame_index,
                timestamp_seconds=self.frame_index / self.info.fps if self.info else None,
                generate_horizontal_flip=generate_horizontal_flip,
                roi_xyxy=format_roi(self.roi),
            )
        except Exception as exc:  # noqa: BLE001 - show filesystem/runtime errors in UI
            messagebox.showerror("儲存失敗", str(exc))
            return False
        self._set_status(
            f"已儲存 {image_path.name} + {label_path.name}；"
            f"{len(self.polygons)} 個 polygon。"
            + ("已同時生成左右翻轉副本。" if generate_horizontal_flip else "")
            + "空白 polygon 表示確認為無煙。"
        )
        if show_message:
            messagebox.showinfo("儲存完成", f"已輸出至：\n{image_path.parent}")
        self.polygons = []
        self.current_polygon = []
        self._render()
        return True

    def _render(self) -> None:
        self.canvas.delete("all")
        if self.frame is None:
            self.canvas.create_text(
                self.canvas.winfo_width() / 2,
                self.canvas.winfo_height() / 2,
                text="請選擇並開啟影片",
                fill="#d8dee9",
                font=("Segoe UI", 16),
            )
            return
        height, width = self.frame.shape[:2]
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        self.display_scale = min(canvas_width / width, canvas_height / height)
        display_width = max(1, int(width * self.display_scale))
        display_height = max(1, int(height * self.display_scale))
        self.display_size = (display_width, display_height)
        self.display_offset = (
            (canvas_width - display_width) / 2,
            (canvas_height - display_height) / 2,
        )
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(
            self.display_offset[0],
            self.display_offset[1],
            image=self.photo,
            anchor="nw",
        )
        for polygon in self.polygons:
            self._draw_polygon(polygon, "#55efc4")
        if self.current_polygon:
            self._draw_polygon(self.current_polygon, "#ffeaa7")

    def _draw_polygon(self, polygon, color: str) -> None:
        points = [coordinate for point in polygon for coordinate in self._to_canvas(point)]
        if len(points) >= 4:
            self.canvas.create_line(*points, fill=color, width=2, smooth=True)
        if len(polygon) >= 3:
            first_x, first_y = self._to_canvas(polygon[0])
            last_x, last_y = self._to_canvas(polygon[-1])
            self.canvas.create_line(last_x, last_y, first_x, first_y, fill=color, width=2)
        for point in polygon:
            x, y = self._to_canvas(point)
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline=color)

    def _set_status(self, value: str) -> None:
        self.status_var.set(value)

    def _close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=None)
    parser.add_argument("--output", default="data/processed/video_yolo")
    parser.add_argument("--weights", default=None, help="Local YOLO11-seg .pt for optional pre-annotation")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument(
        "--roi",
        default="",
        help="Optional ROI in original video coordinates: x1,y1,x2,y2",
    )
    args = parser.parse_args()
    app = AnnotatorApp(args.video, args.output, args.weights, args.split, args.roi)
    app.mainloop()


if __name__ == "__main__":
    main()
