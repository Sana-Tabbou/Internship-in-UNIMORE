import tkinter as tk
from tkinter import filedialog, messagebox
from ttkbootstrap import Style
import ttkbootstrap as tb
import os
import random
import librosa
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.signal import butter, lfilter, hilbert

# -------------------- Traductions --------------------
translations = {
    "en": {
        "title": "🩺 Medical Auscultation Tool",
        "load": "📁 Load folder",
        "random": "🎲 Random audio",
        "play_original": "▶️ Play original",
        "process": "🛠 Signal processing",
        "play_filtered": "▶️ Play filtered",
        "plot_tfr": "📊 Show TFR",
        "spectrum": "📈 Audio Spectrum",
        "classify": "🧠 Classify",
        "reset": "🔄 Reset",
        "choose_processing": "Choose a processing",
        "choose_tfr": "Choose a TFR",
        "lowpass": "Low-pass filter",
        "highpass": "High-pass filter",
        "hpss": "H-P Separation",
        "vme": "VME",
        "cutoff_prompt": "Enter cutoff frequency (Hz)",
        "classification_prompt": "Enter your classification",
        "correct": "✅ Correct!",
        "wrong": "❌ Incorrect.",
        "no_file": "No file loaded",
        "file_selected": "Selected file:",
        "reinitialized": "Reset.",
        "no_wav": "No .wav files in folder",
        "audio_first": "Please load an audio file first.",
        "processing_first": "Please apply processing first.",
        "your_label": "Your label",
        "ground_truth": "Ground truth",
        "result": "Result",
        "select_lang": "🌍 Language"
    },
    "it": {
        "title": "🩺 Strumento di Auscultazione Medica",
        "load": "📁 Carica cartella",
        "random": "🎲 Audio casuale",
        "play_original": "▶️ Riproduci originale",
        "process": "🛠 Elaborazione segnale",
        "play_filtered": "▶️ Riproduci filtrato",
        "plot_tfr": "📊 Mostra TFR",
        "spectrum": "📈 Spettro audio",
        "classify": "🧠 Classifica",
        "reset": "🔄 Reimposta",
        "choose_processing": "Scegli un'elaborazione",
        "choose_tfr": "Scegli una TFR",
        "lowpass": "Filtro passa basso",
        "highpass": "Filtro passa alto",
        "hpss": "Separazione armonica-percussiva",
        "vme": "VME",
        "cutoff_prompt": "Inserisci la frequenza di taglio (Hz)",
        "classification_prompt": "Inserisci la tua classificazione",
        "correct": "✅ Corretto!",
        "wrong": "❌ Errato.",
        "no_file": "Nessun file caricato",
        "file_selected": "File selezionato:",
        "reinitialized": "Reimpostato.",
        "no_wav": "Nessun file .wav nella cartella",
        "audio_first": "Carica prima un file audio.",
        "processing_first": "Applica prima un'elaborazione.",
        "your_label": "La tua etichetta",
        "ground_truth": "Verità di base",
        "result": "Risultato",
        "select_lang": "🌍 Lingua"
    }
}

# -------------------- Traitement --------------------
def butter_filter(data, cutoff, fs, btype, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return lfilter(b, a, data)

def hpss(data):
    y_harmonic, y_percussive = librosa.effects.hpss(data)
    return y_harmonic

def vme(data):
    return librosa.effects.preemphasis(data)

def play_audio(data, sr):
    sd.play(data, sr)
    sd.wait()

def plot_signal_spectrum(data, fs, title="Spectrum"):
    plt.figure(figsize=(10, 4))
    freqs = np.fft.rfftfreq(len(data), d=1/fs)
    fft_spectrum = np.abs(np.fft.rfft(data))
    plt.semilogy(freqs, fft_spectrum)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.tight_layout()
    plt.show()

# -------------------- Interface --------------------
class AudioGUI:
    def __init__(self, master):
        self.master = master
        self.language = "en"
        self.text = translations[self.language]

        master.title(self.text["title"])
        master.geometry("550x500")
        style = Style(theme="cosmo")

        self.dataset_path = ""
        self.file_list = []
        self.audio_data = None
        self.sr = None
        self.current_file = None
        self.filtered_data = None
        self.ground_truth = {}

        self.title_label = tb.Label(master, text=self.text["title"], font=("Segoe UI", 22, "bold"), bootstyle="danger")
        self.title_label.pack(pady=20)
        self.status_label = tb.Label(master, text=self.text["no_file"], font=("Segoe UI", 12))
        self.status_label.pack(pady=5)

        # Langue
        lang_frame = tb.Frame(master)
        lang_frame.pack(pady=5)
        tb.Label(lang_frame, text=self.text["select_lang"]).pack(side="left")
        self.lang_menu = tb.Combobox(lang_frame, values=["English", "Italiano"], width=10)
        self.lang_menu.set("English")
        self.lang_menu.pack(side="left", padx=10)
        self.lang_menu.bind("<<ComboboxSelected>>", self.change_language)

        self.frame = tb.Frame(master)
        self.frame.pack(pady=20, padx=30, fill="x")

        self.buttons = []
        actions = [
            ("load", self.load_dataset),
            ("random", self.pick_random_audio),
            ("play_original", self.play_original_audio),
            ("process", self.signal_processing),
            ("play_filtered", self.play_filtered_audio),
            ("plot_tfr", self.plot_tfr),
            ("spectrum", self.show_spectrum),
            ("classify", self.classify_audio),
            ("reset", self.reset_interface)
        ]

        for key, cmd in actions:
            btn = tb.Button(self.frame, text=self.text[key], command=cmd, bootstyle="danger pill", width=30)
            btn.pack(pady=7)
            self.buttons.append((btn, key))

    def change_language(self, event=None):
        self.language = "it" if self.lang_menu.get() == "Italiano" else "en"
        self.text = translations[self.language]
        self.title_label.config(text=self.text["title"])
        self.status_label.config(text=self.text["no_file"])
        for btn, key in self.buttons:
            btn.config(text=self.text[key])
        self.master.title(self.text["title"])

    def update_status(self, msg):
        self.status_label.config(text=msg)

    def reset_interface(self):
        self.audio_data = None
        self.filtered_data = None
        self.current_file = None
        self.update_status(self.text["reinitialized"])

    def load_dataset(self):
        self.dataset_path = filedialog.askdirectory(title=self.text["load"])
        self.file_list = [f for f in os.listdir(self.dataset_path) if f.lower().endswith('.wav')]
        if not self.file_list:
            messagebox.showwarning("⚠️", self.text["no_wav"])
        else:
            self.update_status(f"{len(self.file_list)} .wav files loaded.")

    def pick_random_audio(self):
        if not self.file_list:
            messagebox.showwarning("⚠️", self.text["no_file"])
            return
        self.current_file = random.choice(self.file_list)
        filepath = os.path.join(self.dataset_path, self.current_file)
        self.audio_data, self.sr = librosa.load(filepath, sr=None)
        self.filtered_data = self.audio_data.copy()

        gt_path = self.current_file.replace('.wav', '.txt')
        if os.path.exists(os.path.join(self.dataset_path, gt_path)):
            with open(os.path.join(self.dataset_path, gt_path)) as f:
                self.ground_truth[self.current_file] = f.read().strip()

        self.update_status(f"{self.text['file_selected']} {self.current_file}")

    def play_original_audio(self):
        if self.audio_data is not None:
            play_audio(self.audio_data, self.sr)

    def play_filtered_audio(self):
        if self.filtered_data is not None:
            play_audio(self.filtered_data, self.sr)

    def ask_cutoff(self):
        return float(tb.inputbox(self.text["cutoff_prompt"], "", parent=self.master))

    def signal_processing(self):
        if self.audio_data is None:
            messagebox.showwarning("⚠️", self.text["audio_first"])
            return

        def apply(choice):
            try:
                if choice in [self.text["lowpass"], self.text["highpass"]]:
                    cutoff = self.ask_cutoff()
                    btype = 'low' if choice == self.text["lowpass"] else 'high'
                    self.filtered_data = butter_filter(self.audio_data, cutoff, self.sr, btype)
                elif choice == self.text["hpss"]:
                    self.filtered_data = hpss(self.audio_data)
                elif choice == self.text["vme"]:
                    self.filtered_data = vme(self.audio_data)
                self.update_status(f"{choice} ✔️")
                top.destroy()
            except Exception as e:
                messagebox.showerror("Error", str(e))

        top = tb.Toplevel(self.master)
        top.title(self.text["process"])
        top.geometry("400x300")
        tb.Label(top, text=self.text["choose_processing"], font=("Segoe UI", 14)).pack(pady=20)

        for txt in [self.text["lowpass"], self.text["highpass"], self.text["hpss"], self.text["vme"]]:
            tb.Button(top, text=txt, bootstyle="danger pill", command=lambda c=txt: apply(c)).pack(pady=10, padx=30, fill="x")

    def plot_tfr(self):
        if self.filtered_data is None:
            messagebox.showwarning("⚠️", self.text["processing_first"])
            return

        def display(choice):
            plt.figure(figsize=(10, 4))
            if choice == "STFT":
                D = librosa.amplitude_to_db(np.abs(librosa.stft(self.filtered_data)), ref=np.max)
                img = librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='log')
                plt.title("STFT")
                plt.colorbar(img)
            elif choice == "Mel":
                S = librosa.feature.melspectrogram(y=self.filtered_data, sr=self.sr)
                S_dB = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_dB, sr=self.sr, x_axis='time', y_axis='mel')
                plt.title("Mel Spectrogram")
                plt.colorbar(img)
            elif choice == "HHT":
                analytic = hilbert(self.filtered_data)
                plt.plot(np.abs(analytic))
                plt.title("HHT (Amplitude)")
            plt.tight_layout()
            plt.show()
            top.destroy()

        top = tb.Toplevel(self.master)
        top.title(self.text["plot_tfr"])
        top.geometry("400x250")
        tb.Label(top, text=self.text["choose_tfr"], font=("Segoe UI", 14)).pack(pady=20)

        for tfr in ["STFT", "Mel", "HHT"]:
            tb.Button(top, text=tfr, bootstyle="danger pill", command=lambda c=tfr: display(c)).pack(pady=10, padx=30, fill="x")

    def show_spectrum(self):
        if self.audio_data is None:
            messagebox.showwarning("⚠️", self.text["audio_first"])
            return
        plot_signal_spectrum(self.audio_data, self.sr, title="Original Signal Spectrum")
        if self.filtered_data is not None:
            plot_signal_spectrum(self.filtered_data, self.sr, title="Filtered Signal Spectrum")

    def classify_audio(self):
        if self.current_file is None:
            messagebox.showwarning("⚠️", self.text["no_file"])
            return
        label = tb.inputbox(self.text["classification_prompt"], "", parent=self.master)
        gt = self.ground_truth.get(self.current_file, "Unknown")
        result = self.text["correct"] if label == gt else self.text["wrong"]
        messagebox.showinfo(self.text["result"], f"{self.text['your_label']}: {label}\n{self.text['ground_truth']}: {gt}\n{result}")

# -------------------- Lancement --------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = AudioGUI(root)
    root.mainloop()
