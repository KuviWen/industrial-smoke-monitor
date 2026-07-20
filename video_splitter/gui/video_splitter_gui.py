"""Tkinter GUI for previewing and splitting local video files."""

from __future__ import annotations

import argparse
import queue
import sys
from pathlib import Path
from threading import Event, Thread

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

CHILD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHILD_ROOT / "src"))

from video_splitter.core import (  # noqa: E402
    build_segments,
    format_timecode,
    parse_timecode,
    read_video_info,
    safe_stem,
    split_video,
    SUPPORTED_VIDEO_EXTENSIONS,
)


class VideoSplitterApp(tk.Tk):
    """Preview a local video and write one or more re-encoded clips."""

    CANVAS_WIDTH = 960
    CANVAS_HEIGHT = 540

    def __init__(self, video: str | None = None, output: str = "video_splitter/data/output") -> None:
        super().__init__()
        self.title("Industrial Smoke Video Splitter")
        self.geometry("1160x850")
        self.minsize(920, 700)

        self.video_path: Path | None = None
        self.info = None
        self.capture: cv2.VideoCapture | None = None
        self.frame = None
        self.frame_index = 0
        self.photo: ImageTk.PhotoImage | None = None
        self.display_scale = 1.0
        self.display_offset = (0.0, 0.0)

        self.worker_thread: Thread | None = None
        self.cancel_event: Event | None = None
        self.worker_queue: queue.Queue = queue.Queue()
        self.closing = False
        self.slider_updating = False

        self.video_var = tk.StringVar(value=video or "")
        self.output_var = tk.StringVar(value=output)
        self.prefix_var = tk.StringVar(value="")
        self.start_var = tk.StringVar(value="00:00:00.000")
        self.end_var = tk.StringVar(value="00:00:00.000")
        self.frame_var = tk.StringVar(value="Frame: —")
        self.time_var = tk.StringVar(value="Time: —")
        self.info_var = tk.StringVar(value="尚未開啟影片")
        self.status_var = tk.StringVar(value="選擇影片後即可預覽與設定切割範圍。")
        self.mode_var = tk.StringVar(value="single")
        self.clip_duration_var = tk.StringVar(value="60")
        self.step_var = tk.StringVar(value="60")
        self.overwrite_var = tk.BooleanVar(value=False)
        self.progress_var = tk.DoubleVar(value=0.0)

        self._build_widgets()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(100, self._poll_worker)
        if video:
            self.after(100, lambda: self._open_video(Path(video)))

    def _build_widgets(self) -> None:
        settings = ttk.Frame(self, padding=8)
        settings.pack(fill="x")
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="輸入影片").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(settings, textvariable=self.video_var).grid(row=0, column=1, columnspan=3, sticky="ew", pady=3)
        ttk.Button(settings, text="選擇影片", command=self._choose_video).grid(row=0, column=4, padx=4)
        ttk.Button(settings, text="開啟", command=self._open_video_from_entry).grid(row=0, column=5)

        ttk.Label(settings, text="輸出資料夾").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(settings, textvariable=self.output_var).grid(row=1, column=1, columnspan=3, sticky="ew", pady=3)
        ttk.Button(settings, text="選擇資料夾", command=self._choose_output).grid(row=1, column=4, padx=4)
        ttk.Label(settings, text="檔名前綴").grid(row=1, column=5, sticky="e", padx=(8, 4))
        ttk.Entry(settings, textvariable=self.prefix_var, width=18).grid(row=1, column=6, sticky="e")

        ttk.Label(settings, textvariable=self.info_var).grid(
            row=2, column=0, columnspan=7, sticky="w", pady=(5, 0)
        )

        preview_frame = ttk.Frame(self, padding=(8, 0, 8, 0))
        preview_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            preview_frame,
            width=self.CANVAS_WIDTH,
            height=self.CANVAS_HEIGHT,
            background="#161a1d",
            highlightthickness=1,
            highlightbackground="#59636e",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._render())

        timeline = ttk.Frame(self, padding=(8, 4, 8, 0))
        timeline.pack(fill="x")
        self.slider = tk.Scale(
            timeline,
            from_=0,
            to=1,
            orient="horizontal",
            showvalue=False,
            resolution=1,
            command=self._on_slider,
            highlightthickness=0,
        )
        self.slider.pack(fill="x")
        status_line = ttk.Frame(timeline)
        status_line.pack(fill="x")
        ttk.Label(status_line, textvariable=self.frame_var).pack(side="left")
        ttk.Label(status_line, textvariable=self.time_var).pack(side="left", padx=20)
        ttk.Button(status_line, text="上一影格", command=lambda: self._move_frame(-1)).pack(side="right", padx=2)
        ttk.Button(status_line, text="下一影格", command=lambda: self._move_frame(1)).pack(side="right", padx=2)
        ttk.Label(status_line, text="跳到影格").pack(side="right", padx=(14, 4))
        self.frame_entry = ttk.Entry(status_line, width=10)
        self.frame_entry.pack(side="right")
        self.frame_entry.bind("<Return>", lambda _event: self._jump_frame())
        ttk.Button(status_line, text="跳到", command=self._jump_frame).pack(side="right", padx=2)

        range_frame = ttk.LabelFrame(self, text="切割範圍", padding=8)
        range_frame.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(range_frame, text="開始時間").grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Entry(range_frame, textvariable=self.start_var, width=17).grid(row=0, column=1, padx=4)
        ttk.Button(range_frame, text="用目前影格設定開始", command=self._set_start).grid(row=0, column=2, padx=4)
        ttk.Label(range_frame, text="結束時間").grid(row=0, column=3, sticky="w", padx=(16, 4))
        ttk.Entry(range_frame, textvariable=self.end_var, width=17).grid(row=0, column=4, padx=4)
        ttk.Button(range_frame, text="用目前影格設定結束", command=self._set_end).grid(row=0, column=5, padx=4)

        mode_frame = ttk.LabelFrame(self, text="輸出模式", padding=8)
        mode_frame.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Radiobutton(
            mode_frame,
            text="單段：輸出一個起訖範圍",
            variable=self.mode_var,
            value="single",
            command=self._update_mode_controls,
        ).grid(row=0, column=0, sticky="w", padx=(0, 18))
        ttk.Radiobutton(
            mode_frame,
            text="批次：依固定長度切成多段",
            variable=self.mode_var,
            value="batch",
            command=self._update_mode_controls,
        ).grid(row=0, column=1, sticky="w", padx=(0, 18))
        ttk.Label(mode_frame, text="每段秒數").grid(row=0, column=2, padx=(0, 4))
        self.clip_entry = ttk.Entry(mode_frame, textvariable=self.clip_duration_var, width=8)
        self.clip_entry.grid(row=0, column=3, padx=(0, 14))
        ttk.Label(mode_frame, text="段與段起點間隔秒數").grid(row=0, column=4, padx=(0, 4))
        self.step_entry = ttk.Entry(mode_frame, textvariable=self.step_var, width=8)
        self.step_entry.grid(row=0, column=5, padx=(0, 14))
        ttk.Label(mode_frame, text="間隔小於每段長度會產生重疊片段").grid(row=0, column=6, sticky="w")

        action_frame = ttk.Frame(self, padding=8)
        action_frame.pack(fill="x")
        self.overwrite_check = ttk.Checkbutton(action_frame, text="允許覆寫同名檔案", variable=self.overwrite_var)
        self.overwrite_check.pack(side="left")
        self.progress = ttk.Progressbar(action_frame, variable=self.progress_var, maximum=1.0, length=300)
        self.progress.pack(side="left", fill="x", expand=True, padx=15)
        self.cancel_button = ttk.Button(action_frame, text="取消", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="right", padx=(4, 0))
        self.split_button = ttk.Button(action_frame, text="開始切割", command=self._start_split)
        self.split_button.pack(side="right")

        footer = ttk.Frame(self, padding=(8, 0, 8, 8))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status_var, wraplength=1050).pack(anchor="w")

        self._update_mode_controls()

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
        path = filedialog.askdirectory(title="選擇輸出資料夾")
        if path:
            self.output_var.set(path)

    def _open_video_from_entry(self) -> None:
        value = self.video_var.get().strip()
        if value:
            self._open_video(Path(value))

    def _open_video(self, path: Path) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("切割進行中", "請先等待目前工作完成或按取消。")
            return
        path = path.expanduser().resolve()
        try:
            info = read_video_info(path)
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError("OpenCV 無法開啟影片")
        except Exception as exc:  # noqa: BLE001 - show UI-friendly runtime errors
            messagebox.showerror("開啟影片失敗", str(exc))
            return
        if self.capture is not None:
            self.capture.release()
        self.video_path = path
        self.video_var.set(str(path))
        self.info = info
        self.capture = capture
        self.frame_index = 0
        self.start_var.set(format_timecode(0))
        self.end_var.set(format_timecode(info.duration_seconds))
        self.prefix_var.set(safe_stem(path.stem))
        self.slider.configure(to=max(1, info.frame_count - 1))
        self._load_frame(0)
        self.info_var.set(
            f"{path.name} | {info.width}x{info.height} | {info.fps:.3f} FPS | "
            f"{info.frame_count} frames | {format_timecode(info.duration_seconds)}"
        )
        self._set_status("影片已開啟；可拖曳時間軸或輸入影格索引預覽。")

    def _load_frame(self, index: int) -> None:
        if self.capture is None or self.info is None:
            return
        index = max(0, min(int(index), self.info.frame_count - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok:
            self._set_status(f"無法讀取影格 {index}")
            return
        self.frame_index = index
        self.frame = frame
        self.frame_var.set(f"Frame: {index} / {self.info.frame_count - 1}")
        self.time_var.set(f"Time: {format_timecode(index / self.info.fps)}")
        self.frame_entry.delete(0, tk.END)
        self.frame_entry.insert(0, str(index))
        self.slider_updating = True
        self.slider.set(index)
        self.slider_updating = False
        self._render()

    def _on_slider(self, value: str) -> None:
        if not self.slider_updating:
            self._load_frame(int(float(value)))

    def _move_frame(self, delta: int) -> None:
        if self.info is not None:
            self._load_frame(self.frame_index + delta)

    def _jump_frame(self) -> None:
        try:
            index = int(self.frame_entry.get())
        except ValueError:
            self._set_status("影格索引必須是整數。")
            return
        self._load_frame(index)

    def _set_start(self) -> None:
        if self.info is not None:
            self.start_var.set(format_timecode(self.frame_index / self.info.fps))

    def _set_end(self) -> None:
        if self.info is not None:
            end = min(self.info.duration_seconds, (self.frame_index + 1) / self.info.fps)
            self.end_var.set(format_timecode(end))

    def _update_mode_controls(self) -> None:
        state = "normal" if self.mode_var.get() == "batch" else "disabled"
        self.clip_entry.configure(state=state)
        self.step_entry.configure(state=state)

    def _parse_segments(self):
        if self.info is None:
            raise ValueError("請先開啟影片")
        start = parse_timecode(self.start_var.get())
        end = parse_timecode(self.end_var.get())
        clip_duration = parse_timecode(self.clip_duration_var.get())
        step = parse_timecode(self.step_var.get())
        return build_segments(
            self.info,
            mode=self.mode_var.get(),
            start_seconds=start,
            end_seconds=end,
            clip_duration_seconds=clip_duration,
            step_seconds=step,
        )

    def _start_split(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        try:
            segments = self._parse_segments()
            output_dir = self.output_var.get().strip()
            if not output_dir:
                raise ValueError("請指定輸出資料夾")
            prefix = self.prefix_var.get().strip() or (self.video_path.stem if self.video_path else "video")
            video_path = self.video_path
            if video_path is None:
                raise ValueError("請先開啟影片")
        except Exception as exc:  # noqa: BLE001 - validation belongs in the GUI
            messagebox.showerror("切割設定錯誤", str(exc))
            return

        self.progress_var.set(0.0)
        self.cancel_event = Event()
        overwrite = self.overwrite_var.get()
        self.split_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self._set_status(f"準備切割 {len(segments)} 段；輸出影片不包含音訊軌。")
        self.worker_thread = Thread(
            target=self._worker,
            args=(video_path, output_dir, segments, prefix, overwrite, self.cancel_event),
            daemon=True,
        )
        self.worker_thread.start()

    def _worker(self, video_path, output_dir, segments, prefix, overwrite, cancel_event: Event) -> None:
        try:
            outputs = split_video(
                video_path,
                output_dir,
                segments,
                prefix=prefix,
                overwrite=overwrite,
                progress_callback=lambda progress, message: self.worker_queue.put(
                    ("progress", progress, message)
                ),
                cancel_event=cancel_event,
            )
            self.worker_queue.put(("done", outputs, cancel_event.is_set()))
        except Exception as exc:  # noqa: BLE001 - forward worker error to main thread
            self.worker_queue.put(("error", exc))

    def _poll_worker(self) -> None:
        try:
            while True:
                message = self.worker_queue.get_nowait()
                kind = message[0]
                if kind == "progress":
                    self.progress_var.set(float(message[1]))
                    self._set_status(str(message[2]))
                elif kind == "done":
                    outputs, cancelled = message[1], message[2]
                    self._finish_worker()
                    if cancelled:
                        self._set_status(f"已取消；已完成輸出 {len(outputs)} 個檔案。")
                    else:
                        self.progress_var.set(1.0)
                        self._set_status(f"切割完成，共輸出 {len(outputs)} 個檔案。")
                        messagebox.showinfo(
                            "切割完成",
                            f"已輸出 {len(outputs)} 個檔案至：\n{self.output_var.get()}",
                        )
                elif kind == "error":
                    self._finish_worker()
                    messagebox.showerror("切割失敗", str(message[1]))
                    self._set_status("切割失敗，請查看錯誤訊息並確認輸出資料夾與編碼器。")
        except queue.Empty:
            pass
        if not self.closing:
            self.after(100, self._poll_worker)

    def _finish_worker(self) -> None:
        self.worker_thread = None
        self.cancel_event = None
        self.split_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")

    def _cancel(self) -> None:
        if self.cancel_event is not None:
            self.cancel_event.set()
            self._set_status("正在取消；目前影格寫入完成後會停止。")

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
        self.display_offset = (
            (canvas_width - display_width) / 2,
            (canvas_height - display_height) / 2,
        )
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize(
            (display_width, display_height), Image.Resampling.LANCZOS
        )
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(
            self.display_offset[0],
            self.display_offset[1],
            image=self.photo,
            anchor="nw",
        )

    def _set_status(self, value: str) -> None:
        self.status_var.set(value)

    def _close(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.closing = True
            if self.cancel_event is not None:
                self.cancel_event.set()
            self.after(100, self._close)
            return
        if self.capture is not None:
            self.capture.release()
        self.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=None)
    parser.add_argument("--output", default="video_splitter/data/output")
    args = parser.parse_args()
    app = VideoSplitterApp(args.video, args.output)
    app.mainloop()


if __name__ == "__main__":
    main()
