import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk

BASE = "/storage/emulated/0/OnT"
FOLDERS = ["intrdpcy", "OnT", "AIC", "MathVis", "TrmaVis", "BDTHNGS", "00-inbox"]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def append_to_dump(folder, text):
    path = os.path.join(BASE, folder)
    ensure_dir(path)
    dump = os.path.join(path, "dump.md")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n## {ts} — {folder}\n\n{text.strip()}\n"

    if not os.path.exists(dump):
        with open(dump, "w", encoding="utf-8") as f:
            f.write(f"# {folder} — dump\n")
            f.write(entry)
    else:
        with open(dump, "a", encoding="utf-8") as f:
            f.write(entry)

def quit_app():
    root.quit()
    root.destroy()

root = tk.Tk()
root.title("QuickDump")
root.resizable(False, False)

frame = ttk.Frame(root, padding=8)
frame.pack()

folder = tk.StringVar(value=FOLDERS[0])
status = tk.StringVar(value="Ready.")

top = ttk.Frame(frame)
top.pack(fill="x")

ttk.Combobox(top, values=FOLDERS, textvariable=folder, width=12, state="readonly").pack(side="left", padx=(0,6))
ttk.Button(top, text="Save", command=lambda: save()).pack(side="left", padx=(0,6))
ttk.Button(top, text="Exit", command=quit_app).pack(side="left")

text = tk.Text(frame, width=44, height=6)
text.pack(pady=(6,6))

bottom = ttk.Frame(frame)
bottom.pack(fill="x")

def save():
    content = text.get("1.0", "end").strip()
    if not content:
        status.set("Nothing to save.")
        return
    append_to_dump(folder.get(), content)
    text.delete("1.0", "end")
    status.set("Saved.")

ttk.Button(bottom, text="Clear", command=lambda: text.delete("1.0","end")).pack(side="left", padx=(0,6))
ttk.Label(bottom, textvariable=status).pack(side="left")

root.mainloop()