import threading
import time
import random
import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import ImageGrab, Image
import numpy as np
import cv2
import pytesseract  # optional; used if template not found
import os
import sys

# ---------- Настройки (можешь менять) ----------
CENTER_REGION_W = 420   # ширина зоны захвата центра (пикс)
CENTER_REGION_H = 220   # высота зоны захвата центра
REGION_OFFSET_Y = 0     # смещение по Y от центра (положительное - вниз)
TEMPLATE_DIR_WORDS = "words"   # папка со скриншотами STAMINA, ...
TEMPLATE_DIR_KEYS = "keys"     # папка со скриншотами Q W E ...
LOGFILE = "auto_macro_log.txt"
USE_TESSERACT_AS_FALLBACK = True
# ------------------------------------------------

# Helper: get center region coords (left,top,right,bottom)
def get_center_region():
    screen = ImageGrab.grab()
    sw, sh = screen.size
    cx, cy = sw // 2, sh // 2 + REGION_OFFSET_Y
    left = max(0, cx - CENTER_REGION_W // 2)
    top = max(0, cy - CENTER_REGION_H // 2)
    right = min(sw, cx + CENTER_REGION_W // 2)
    bottom = min(sh, cy + CENTER_REGION_H // 2)
    return (left, top, right, bottom)

def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    app_append_log(line)

# GUI will call this to append logs
gui_textbox = None
def app_append_log(text):
    global gui_textbox
    if gui_textbox:
        gui_textbox.configure(state='normal')
        gui_textbox.insert(tk.END, text + "\n")
        gui_textbox.see(tk.END)
        gui_textbox.configure(state='disabled')

# Template matching utility
def load_templates_from_dir(d):
    templates = {}
    if not os.path.isdir(d):
        return templates
    for fname in os.listdir(d):
        if not fname.lower().endswith(('.png','.jpg','.jpeg','bmp')):
            continue
        key = os.path.splitext(fname)[0].upper()
        img = cv2.imread(os.path.join(d, fname), cv2.IMREAD_UNCHANGED)
        if img is not None:
            templates[key] = img
    return templates

def match_template(region_cv, templates, threshold=0.75):
    """Return best match (key, max_val, max_loc, w,h) or (None,0,None,0,0)"""
    best = (None, 0, None, 0, 0)
    for key, tpl in templates.items():
        try:
            tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) if tpl.ndim==3 else tpl
            reg_gray = cv2.cvtColor(region_cv, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(reg_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            minv, maxv, minloc, maxloc = cv2.minMaxLoc(res)
            if maxv > best[1]:
                best = (key, float(maxv), maxloc, tpl.shape[1], tpl.shape[0])
        except Exception as e:
            # skip template if mismatch
            continue
    if best[1] >= threshold:
        return best
    return (None, best[1], None, 0, 0)

def ocr_read_text(region_pil):
    # Preprocess image to improve OCR
    img = region_pil.convert('L')  # gray
    arr = np.array(img)
    # simple thresh
    _, arr = cv2.threshold(arr, 180, 255, cv2.THRESH_BINARY)
    img2 = Image.fromarray(arr)
    text = pytesseract.image_to_string(img2, config='--psm 7').strip()
    return text.upper()

# Main worker thread
class Worker(threading.Thread):
    def _init_(self, gui):
        super()._init_()
        self.gui = gui
        self.running = False
        self._stop = threading.Event()
        self.word_templates = load_templates_from_dir(TEMPLATE_DIR_WORDS)
        self.key_templates = load_templates_from_dir(TEMPLATE_DIR_KEYS)
    def start_cycle(self):
        if not self.running:
            self._stop.clear()
            self.running = True
            if not self.is_alive():
                self.start()
    def stop_cycle(self):
        self._stop.set()
        self.running = False
    def run(self):
        log("Worker loop started.")
        cycle_count = 0
        while not self._stop.is_set():
            cycle_count += 1
            # 1) Инструкция: нажать E один раз (не автоматизируем)
            log("Instruction: Press key E once now.")
            self.gui.set_status("Press E (once) — please press manually")
            # optional: wait a little for user to press E
            time.sleep(0.6)

            # capture center region
            left, top, right, bottom = get_center_region()
            region = ImageGrab.grab(bbox=(left, top, right, bottom))
            region_cv = pil_to_cv(region)

            # 2) распознать слово из списка
            target_word = self.gui.stat_choice.get().upper()
            found_word = None
            # If templates exist, try template matching first
            if self.word_templates:
                match_key, score, loc, w, h = match_template(region_cv, self.word_templates, threshold=0.72)
                if match_key:
                    found_word = match_key
                    log(f"Found word candidate by template: {found_word} (score={score:.2f})")
                else:
                    log(f"No strong template match for words (best score {score:.2f}).")
            # fallback to OCR if enabled
            if not found_word and USE_TESSERACT_AS_FALLBACK:
                try:
                    txt = ocr_read_text(region)
                    log(f"OCR read: '{txt}'")
                    # find if any of the known words present in OCR text
                    for wname in ['STAMINA','STRENGTH','AGILITY','DURABILITY','MUSCLE']:
                        if wname in txt:
                            found_word = wname
                            break
                except Exception as e:
                    log("OCR failed: " + str(e))

            # Show result and prompt
            if found_word:
                log(f"Detected word: {found_word}")
                if found_word == target_word:
                    self.gui.set_status(f"Target '{target_word}' detected — please move mouse to it and LEFT CLICK")
                    # Give user time to move mouse and click
                    # We can show a countdown to let user click
                    for i in range(5,0,-1):
                        if self._stop.is_set(): break
                        self.gui.set_status(f"Move cursor & left-click target now — {i}s")
                        time.sleep(1)
                    log("Please ensure you clicked on the chosen stat (manual action).")
                else:
                    log(f"Detected word '{found_word}' does not match target '{target_word}'. Will continue.")
                    self.gui.set_status(f"Detected '{found_word}', not target — continuing.")
                    time.sleep(0.8)
            else:
                log("No word detected in center region.")
                self.gui.set_status("No stat detected — make sure UI is open and centered.")
                time.sleep(0.8)

            # 3) распознать букву (Q..C) в центре
            # capture again because second image shows the key
            region2 = ImageGrab.grab(bbox=(left, top, right, bottom))
            region2_cv = pil_to_cv(region2)
            found_key = None
            if self.key_templates:
                match_key, score, loc, w, h = match_template(region2_cv, self.key_templates, threshold=0.70)
                if match_key:
                    found_key = match_key
                    log(f"Found key by template: {found_key} (score={score:.2f})")
            if not found_key and USE_TESSERACT_AS_FALLBACK:
                try:
                    txt2 = ocr_read_text(region2)
                    log(f"OCR read for key: '{txt2}'")
                    # single-character search
                    for ch in ['Q','W','E','A','S','D','Z','X','C']:
                        if ch in txt2:
                            found_key = ch
                            break
                except Exception as e:
                    log("OCR for key failed: " + str(e))

            if found_key:
                log(f"Detected key: {found_key}")
                self.gui.set_status(f"Detected key: {found_key} — you should press it manually as instructed.")
            else:
                log("No key/letter detected.")
                self.gui.set_status("No key detected — ensure training bar is visible.")
                time.sleep(0.7)

            # 4) Инструкция: нажимать распознанную клавишу вручную в течении времени
            hold_seconds = float(self.gui.entry_time.get())
            interval_ms = float(self.gui.entry_interval.get())
            if found_key:
                # Inform user and countdown
                log(f"Start manual pressing: Key {found_key} for {hold_seconds}s, interval {interval_ms}ms.")
                self.gui.set_status(f"Manual step: Press and release '{found_key}' repeatedly for {int(hold_seconds)}s.")
                t_end = time.time() + hold_seconds
                # visual countdown loop
                while time.time() < t_end and not self._stop.is_set():
                    remaining = int(t_end - time.time())
                    self.gui.set_progress((hold_seconds - (t_end - time.time())) / hold_seconds * 100, f"Pressing {found_key}: {remaining}s left")
                    time.sleep(0.5)
                log("Manual pressing period finished (user should stop pressing key).")
            else:
                log("Skipping pressing step (no key detected).")

            # 5) Инструкция: зажать W на 2 секунды (manual)
            log("Instruction: Hold W for 2 seconds (manual).")
            self.gui.set_status("Please HOLD 'W' for 2 seconds now.")
            for i in range(2,0,-1):
                if self._stop.is_set(): break
                self.gui.set_progress(100, f"Holding W: {i}s")
                time.sleep(1)
            self.gui.set_progress(0, "")
            log("Hold-W step done.")

            # end of cycle
            log(f"Cycle {cycle_count} finished.")
            self.gui.set_status("Cycle complete. Waiting 0.8s before next.")
            time.sleep(0.8)
        self.running = False
        log("Worker loop stopped.")
        self.gui.set_status("Stopped")

# Simple GUI class
class AppGUI:
    def _init_(self, root):
        self.root = root
        root.title("Fistborn — Recognition & Manual Assistant (SAFE)")
        # top controls
        frm = ttk.Frame(root, padding=8)
        frm.pack(fill=tk.X)

        ttk.Label(frm, text="Target Stat:").grid(row=0, column=0, sticky=tk.W)
        self.stat_choice = ttk.Combobox(frm, values=['STAMINA','STRENGTH','AGILITY','DURABILITY','MUSCLE'], state='readonly')
        self.stat_choice.current(0)
        self.stat_choice.grid(row=0, column=1, padx=6)

        ttk.Label(frm, text="Press duration (s):").grid(row=0, column=2, sticky=tk.W)
        self.entry_time = ttk.Entry(frm, width=8)
        self.entry_time.insert(0, "60")
        self.entry_time.grid(row=0, column=3, padx=6)

        ttk.Label(frm, text="Interval (ms):").grid(row=0, column=4, sticky=tk.W)
        self.entry_interval = ttk.Entry(frm, width=8)
        self.entry_interval.insert(0, "100")
        self.entry_interval.grid(row=0, column=5, padx=6)

        self.btn_toggle = ttk.Button(frm, text="Start (F6)", command=self.on_toggle)
        self.btn_toggle.grid(row=0, column=6, padx=6)

        ttk.Label(frm, text="Status:").pack(anchor='w', padx=8, pady=(6,0))
        self.status_var = tk.StringVar(value="Idle")
        self.status_lbl = ttk.Label(root, textvariable=self.status_var, background="#fff", relief='sunken')
        self.status_lbl.pack(fill=tk.X, padx=8, pady=(0,6))

        # log
        ttk.Label(root, text="Activity Log:").pack(anchor='w', padx=8)
        global gui_textbox
        gui_textbox = scrolledtext.ScrolledText(root, height=12, state='disabled')
        gui_textbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

        # progress
        self.progress = ttk.Progressbar(root, orient='horizontal', length=400, mode='determinate')
        self.progress.pack(fill=tk.X, padx=8, pady=(0,8))

        # worker
        self.worker = Worker(self)

        # hotkey F6 via tkinter binding (works when window focused)
        root.bind('<F6>', lambda e: self.on_toggle())

        # message about safety
        ttk.Label(root, text="Note: This tool only detects and prompts — it does NOT press keys/clicks automatically.").pack(anchor='w', padx=8, pady=(0,8))

    def on_toggle(self):
        if not self.worker.running:
            # start
            try:
                float(self.entry_time.get())
                float(self.entry_interval.get())
            except:
                messagebox.showerror("Invalid", "Time and interval must be numeric.")
                return
            self.worker = Worker(self)  # recreate to reload templates
            self.worker.start_cycle()
            self.btn_toggle.config(text="Stop (F6)")
            self.set_status("Running")
            log("User started monitoring (F6).")
        else:
            self.worker.stop_cycle()
            self.btn_toggle.config(text="Start (F6)")
            self.set_status("Stopping...")
            log("User requested stop (F6).")

    def set_status(self, text):
        self.status_var.set(text)

    def set_progress(self, percent, text=""):
        try:
            self.progress['value'] = percent
            if text:
                self.set_status(text)
        except:
            pass

if _name_ == "_main_":
    root = tk.Tk()
    app = AppGUI(root)
    root.geometry("780x600")
    root.mainloop()
