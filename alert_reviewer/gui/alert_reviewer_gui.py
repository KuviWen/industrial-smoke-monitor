"""Tkinter GUI for inspecting and relabeling saved smoke alerts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alert_reviewer.review import AlertReviewService  # noqa: E402


class AlertReviewerApp:
    def __init__(self, window, input_dir: Path, dataset_dir: Path, split: str) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.window = window
        self.window.title("Industrial Smoke Monitor - Alert Reviewer")
        self.window.geometry("1400x860")
        self.service = AlertReviewService(input_dir)
        self.dataset_dir = dataset_dir
        self.split = split
        self.items = self.service.list_alerts()
        self.index = 0
        self.original_image: Image.Image | None = None
        self.display_scale = 1.0
        self.display_offset = (0.0, 0.0)
        self.polygons: list[list[tuple[float, float]]] = []
        self.current_polygon: list[tuple[float, float]] = []

        self.input_var = tk.StringVar(value=str(input_dir))
        self.dataset_var = tk.StringVar(value=str(dataset_dir))
        self.split_var = tk.StringVar(value=split)
        self.label_var = tk.StringVar(value="smoke")
        self.action_var = tk.StringVar(value="保留在原地")
        self.status_var = tk.StringVar()
        self._build()
        self._refresh_listbox()
        self._load_current()

    def _build(self) -> None:
        tk, ttk = self.tk, self.ttk
        self.window.columnconfigure(1, weight=1)
        self.window.rowconfigure(1, weight=1)

        settings = ttk.Frame(self.window, padding=8)
        settings.grid(row=0, column=0, columnspan=3, sticky="ew")
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(4, weight=1)
        ttk.Label(settings, text="告警資料夾").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.input_var).grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=(6, 14)
        )
        ttk.Label(settings, text="訓練資料集").grid(row=0, column=3, sticky="w")
        ttk.Entry(settings, textvariable=self.dataset_var).grid(
            row=0, column=4, sticky="ew", padx=(6, 8)
        )
        ttk.Label(settings, text="split").grid(row=0, column=5, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.split_var,
            values=("train", "val", "test"),
            width=8,
            state="readonly",
        ).grid(row=0, column=6, sticky="w", padx=(6, 8))
        ttk.Button(settings, text="重新載入", command=self.reload).grid(
            row=0, column=7, sticky="e"
        )

        left = ttk.Frame(self.window, padding=(8, 0, 4, 8))
        left.grid(row=1, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.listbox = tk.Listbox(left, width=30, exportselection=False)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        list_scroll = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        list_scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=list_scroll.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        center = ttk.Frame(self.window, padding=(4, 0, 4, 8))
        center.grid(row=1, column=1, sticky="nsew")
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(center, background="#202124", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", lambda _event: self._render())

        right = ttk.Frame(self.window, padding=(4, 0, 8, 8))
        right.grid(row=1, column=2, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text="模型與複核資訊").grid(row=0, column=0, sticky="w")
        self.info = tk.Text(right, width=42, height=18, wrap="word", state="disabled")
        self.info.grid(row=1, column=0, sticky="nsew", pady=(4, 8))

        label_frame = ttk.LabelFrame(right, text="人工分類")
        label_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Radiobutton(
            label_frame, text="smoke（有煙）", variable=self.label_var, value="smoke",
            command=self._render,
        ).pack(anchor="w", padx=8, pady=3)
        ttk.Radiobutton(
            label_frame, text="no smoke（無煙／誤報）", variable=self.label_var,
            value="no_smoke", command=self._render,
        ).pack(anchor="w", padx=8, pady=3)

        draw_frame = ttk.Frame(right)
        draw_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(draw_frame, text="完成目前 polygon", command=self.finish_polygon).pack(
            side="left", padx=(0, 5)
        )
        ttk.Button(draw_frame, text="清除全部線條", command=self.clear_polygons).pack(
            side="left"
        )
        ttk.Label(
            right,
            text="在影像上依序點擊煙霧輪廓；至少 3 點後按完成。\n原模型 polygon 會先顯示，可修改或清除。",
            wraplength=330,
            justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(0, 10))

        action_frame = ttk.LabelFrame(right, text="完成後處理")
        action_frame.grid(row=5, column=0, sticky="ew")
        ttk.Combobox(
            action_frame,
            textvariable=self.action_var,
            values=("保留在原地", "移動到訓練資料集", "刪除"),
            state="readonly",
        ).pack(fill="x", padx=8, pady=(8, 6))
        ttk.Button(action_frame, text="套用目前告警", command=self.apply_action).pack(
            fill="x", padx=8, pady=(0, 8)
        )
        ttk.Button(right, text="上一筆", command=self.previous).grid(
            row=6, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Button(right, text="下一筆", command=self.next).grid(
            row=6, column=0, sticky="e", pady=(10, 0)
        )
        ttk.Label(right, textvariable=self.status_var).grid(
            row=7, column=0, sticky="w", pady=(8, 0)
        )

    def reload(self) -> None:
        self.service = AlertReviewService(Path(self.input_var.get()).expanduser())
        self.dataset_dir = Path(self.dataset_var.get()).expanduser()
        self.split = self.split_var.get()
        self.items = self.service.list_alerts()
        self.index = min(self.index, max(0, len(self.items) - 1))
        self._refresh_listbox()
        self._load_current()

    def _refresh_listbox(self) -> None:
        self.listbox.delete(0, self.tk.END)
        for item in self.items:
            review = item.metadata.get("review", {})
            label = (
                review.get("label", item.metadata.get("classification", "smoke"))
                if isinstance(review, dict)
                else item.metadata.get("classification", "smoke")
            )
            self.listbox.insert(
                self.tk.END,
                f"{item.image_path.name}\n[{item.review_status}] {label}",
            )
        if self.items:
            self.listbox.selection_set(self.index)
            self.listbox.see(self.index)

    def _load_current(self) -> None:
        if not self.items:
            self.original_image = None
            self.polygons = []
            self.current_polygon = []
            self.canvas.delete("all")
            self._set_info("目前沒有找到告警 JSON／JPG。\n請確認輸入資料夾。")
            self.status_var.set("0 筆告警")
            return
        item = self.items[self.index]
        try:
            self.original_image = Image.open(item.image_path).convert("RGB")
        except OSError as exc:
            self._set_info(f"影像讀取失敗：{exc}")
            return
        self.polygons = self.service.initial_polygons(item)
        self.current_polygon = []
        review = item.metadata.get("review", {})
        label = review.get("label") if isinstance(review, dict) else None
        self.label_var.set(label if label in {"smoke", "no_smoke"} else "smoke")
        self._set_info(json.dumps(self._summary(item), ensure_ascii=False, indent=2, default=str))
        self.status_var.set(f"{self.index + 1} / {len(self.items)}")
        self._render()

    @staticmethod
    def _summary(item) -> dict:
        metadata = item.metadata
        return {
            "file": item.image_path.name,
            "timestamp_utc": metadata.get("timestamp_utc"),
            "classification": metadata.get("classification", metadata.get("cls")),
            "conf": metadata.get("conf", metadata.get("max_confidence")),
            "area": metadata.get("area", metadata.get("smoke_area_ratio")),
            "instance_count": metadata.get("instance_count"),
            "roi_xyxy": metadata.get("roi_xyxy"),
            "instances": metadata.get("instances", []),
            "review": metadata.get("review", {"status": "unreviewed"}),
        }

    def _set_info(self, text: str) -> None:
        self.info.configure(state="normal")
        self.info.delete("1.0", self.tk.END)
        self.info.insert("1.0", text)
        self.info.configure(state="disabled")

    def _render(self) -> None:
        self.canvas.delete("all")
        if self.original_image is None:
            return
        canvas_width = max(100, self.canvas.winfo_width())
        canvas_height = max(100, self.canvas.winfo_height())
        image_width, image_height = self.original_image.size
        self.display_scale = min(
            (canvas_width - 20) / image_width,
            (canvas_height - 20) / image_height,
            1.0,
        )
        display_size = (
            max(1, int(image_width * self.display_scale)),
            max(1, int(image_height * self.display_scale)),
        )
        display = self.original_image.resize(display_size, Image.Resampling.LANCZOS)
        self.display_photo = ImageTk.PhotoImage(display)
        offset_x = (canvas_width - display_size[0]) / 2
        offset_y = (canvas_height - display_size[1]) / 2
        self.display_offset = (offset_x, offset_y)
        self.canvas.create_image(offset_x, offset_y, image=self.display_photo, anchor="nw")

        for polygon in self.polygons:
            self._draw_polygon(polygon, "#00ff7f")
        if self.current_polygon:
            self._draw_polygon(self.current_polygon, "#00bfff", close=False)

    def _draw_polygon(self, polygon, color: str, close: bool = True) -> None:
        points = [self._to_canvas(point) for point in polygon]
        if len(points) >= 2:
            flattened = [value for point in points for value in point]
            if close and len(points) >= 3:
                flattened += list(points[0])
            self.canvas.create_line(*flattened, fill=color, width=2, joint="curve")
        for x, y in points:
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="")

    def _to_canvas(self, point: tuple[float, float]) -> tuple[float, float]:
        return (
            self.display_offset[0] + point[0] * self.display_scale,
            self.display_offset[1] + point[1] * self.display_scale,
        )

    def _to_image(self, x: float, y: float) -> tuple[float, float] | None:
        if self.original_image is None or self.display_scale <= 0:
            return None
        image_x = (x - self.display_offset[0]) / self.display_scale
        image_y = (y - self.display_offset[1]) / self.display_scale
        width, height = self.original_image.size
        if not (0 <= image_x < width and 0 <= image_y < height):
            return None
        return image_x, image_y

    def _on_canvas_click(self, event) -> None:
        point = self._to_image(event.x, event.y)
        if point is not None:
            self.current_polygon.append(point)
            self._render()

    def finish_polygon(self) -> None:
        if len(self.current_polygon) < 3:
            self.status_var.set("目前 polygon 至少需要 3 個點")
            return
        self.polygons.append(self.current_polygon)
        self.current_polygon = []
        self._render()

    def clear_polygons(self) -> None:
        self.polygons = []
        self.current_polygon = []
        self._render()

    def _on_list_select(self, _event) -> None:
        selection = self.listbox.curselection()
        if selection and selection[0] != self.index:
            self.index = selection[0]
            self._load_current()

    def previous(self) -> None:
        if self.items:
            self.index = max(0, self.index - 1)
            self._refresh_listbox()
            self._load_current()

    def next(self) -> None:
        if self.items:
            self.index = min(len(self.items) - 1, self.index + 1)
            self._refresh_listbox()
            self._load_current()

    def apply_action(self) -> None:
        if not self.items:
            return
        from tkinter import messagebox

        item = self.items[self.index]
        label = self.label_var.get()
        polygons = [list(polygon) for polygon in self.polygons]
        if len(self.current_polygon) >= 3:
            polygons.append(list(self.current_polygon))
        if label == "smoke" and not polygons:
            if not messagebox.askyesno(
                "沒有 polygon",
                "目前沒有手動 polygon，仍要以 smoke 保存嗎？",
            ):
                return
        action = self.action_var.get()
        try:
            if action == "刪除":
                if not messagebox.askyesno(
                    "確認刪除",
                    "這會刪除告警影像、JSON 與 label，且不會放入訓練資料集。確定嗎？",
                ):
                    return
                self.service.delete(item)
            elif action == "移動到訓練資料集":
                self.service.move_to_dataset(
                    item, label, polygons, self.dataset_dir, self.split_var.get()
                )
            else:
                self.service.keep_in_place(item, label, polygons)
        except Exception as exc:
            messagebox.showerror("處理失敗", str(exc))
            return

        old_index = self.index
        self.items = self.service.list_alerts()
        self.index = min(old_index, max(0, len(self.items) - 1))
        self._refresh_listbox()
        self._load_current()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "runtime" / "alerts",
        help="Alert artifact directory",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "field_yolo",
        help="Parent-compatible YOLO dataset output directory",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    args = parser.parse_args()

    import tkinter as tk

    window = tk.Tk()
    AlertReviewerApp(window, args.input, args.dataset, args.split)
    window.mainloop()


if __name__ == "__main__":
    main()
