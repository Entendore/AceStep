# ==============================================================
# ACE-STEP MUSIC STUDIO - PySide6 Application
# ==============================================================
# A comprehensive GUI for the ACE-Step Music Generation Foundation Model.
# Features: Text2Music, Retake, Repaint, Lyric Editing, Batch Generation,
# Preset Management, Hardware Optimizations, and History Tracking.
# ==============================================================


# ==============================================================
# 1. IMPORTS & DEPENDENCIES
# ==============================================================
import sys
import os
import json
import torch
import numpy as np
import soundfile as sf
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl, QSettings, QTime
from PySide6.QtGui import QColor, QPalette, QIcon, QKeySequence, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QFileDialog, QProgressBar, QMessageBox, QTabWidget,
    QGroupBox, QCheckBox, QSlider, QRadioButton, QDialog, QFormLayout,
    QListWidget, QListWidgetItem, QSplitter, QFrame, QButtonGroup
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput


# ==============================================================
# 2. MOCK BACKEND (ACE-Step Bridge)
# ==============================================================
# Simulates the ACE-Step API for UI testing.
# Replace this class with actual ACE-Step API calls when ready.
class ACEStepBridge:
    def __init__(self):
        self.is_loaded = False

    def load_model(self, device="cuda:0", precision="bf16", model_folder=None, 
                   torch_compile=False, cpu_offload=False, overlapped_decode=False):
        folder_msg = model_folder if model_folder else "~/.cache/ace-step/checkpoints (Default)"
        print(f"[ACE-Step] Loading model on {device} with {precision} from {folder_msg}...")
        print(f"[ACE-Step] Optimizations -> torch_compile: {torch_compile}, cpu_offload: {cpu_offload}, overlapped: {overlapped_decode}")
        
        import time
        time.sleep(1.5)
        self.is_loaded = True
        print("[ACE-Step] Model loaded successfully.")
        return True

    def generate(self, params):
        if not self.is_loaded:
            raise Exception("Model not loaded!")
        print(f"[ACE-Step] Generating with config: Tags='{params.get('tags')}', Lang='{params.get('language')}', Mode='{params.get('app_mode')}'")
        
        steps = params.get('steps', 27)
        for i in range(steps):
            pass # Worker thread handles visual step logging
        
        sample_rate = 44100
        duration = params.get('duration', 30)
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        freq = 440 if "pop" in params.get('tags', '').lower() else 330
        audio_data = 0.4 * np.sin(2 * np.pi * freq * t) * np.exp(-0.05 * t)
        audio_data += 0.1 * np.random.randn(len(t)) 
        
        return audio_data, sample_rate


# ==============================================================
# 3. WORKER THREADS
# ==============================================================
# Handles background processing to prevent UI freezing.
class GenerationWorker(QThread):
    progress = Signal(int)
    step_log = Signal(str)
    track_finished = Signal(object, int, dict)
    batch_finished = Signal()
    error = Signal(str)

    def __init__(self, bridge, params, batch_count=1):
        super().__init__()
        self.bridge = bridge
        self.params = params
        self.batch_count = batch_count

    def run(self):
        try:
            total_steps = self.params.get('steps', 27)
            
            for batch_idx in range(self.batch_count):
                self.step_log.emit(f"--- Starting Generation {batch_idx + 1} of {self.batch_count} ---")
                
                current_params = self.params.copy()
                if self.batch_count > 1:
                    current_params['seed'] = self.params['seed'] + batch_idx
                    self.step_log.emit(f"Using Seed: {current_params['seed']}")

                self.progress.emit(0)
                
                # Simulate Diffusion ODE Steps
                for step in range(total_steps):
                    progress_val = int((step / total_steps) * 100)
                    self.progress.emit(progress_val)
                    self.step_log.emit(f"Denoising step {step+1}/{total_steps}...")
                    import time
                    time.sleep(0.05) 
                    
                audio_data, sample_rate = self.bridge.generate(current_params)
                
                self.progress.emit(100)
                self.step_log.emit("Generation complete. Decoding audio...")
                self.track_finished.emit(audio_data, sample_rate, current_params)
                
            self.batch_finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))


# ==============================================================
# 4. UTILITY CLASSES
# ==============================================================
# Redirects stdout/stderr to the UI log console.
class LogRedirector:
    def __init__(self, signal):
        self.signal = signal
    def write(self, text):
        if text.strip():
            self.signal.emit(text.strip())
    def flush(self):
        pass


# ==============================================================
# 5. DIALOG WINDOWS
# ==============================================================
# --- Settings Dialog ---
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model & Hardware Settings")
        self.setModal(True)
        self.settings = QSettings("ACEStep", "ACEStepApp")
        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda:0", "cuda:1"])
        form.addRow("Device:", self.device_combo)

        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["float32", "bfloat16", "float16"])
        form.addRow("Precision:", self.precision_combo)

        model_folder_layout = QHBoxLayout()
        self.model_folder_edit = QLineEdit()
        self.model_folder_edit.setPlaceholderText("Leave empty for default (~/.cache/ace-step)")
        browse_btn = QPushButton("📁 Browse")
        browse_btn.clicked.connect(self.browse_model_folder)
        clear_btn = QPushButton("✖ Clear")
        clear_btn.clicked.connect(lambda: self.model_folder_edit.clear())
        model_folder_layout.addWidget(self.model_folder_edit)
        model_folder_layout.addWidget(browse_btn)
        model_folder_layout.addWidget(clear_btn)
        form.addRow("Custom Model Folder:", model_folder_layout)

        self.torch_compile = QCheckBox("torch.compile() (Faster inference)")
        self.cpu_offload = QCheckBox("CPU Offload (Saves VRAM)")
        self.overlapped = QCheckBox("Overlapped Decode (Speeds up inference)")
        form.addRow("Optimizations:", self.torch_compile)
        form.addRow("", self.cpu_offload)
        form.addRow("", self.overlapped)

        layout.addLayout(form)

        info_label = QLabel("ℹ️ If the custom folder is empty, the model will automatically download to this location.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #aaaaaa; font-size: 10px; margin-top: 10px;")
        layout.addWidget(info_label)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save & Load")
        save_btn.clicked.connect(self.save_and_accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def browse_model_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Custom Model Folder")
        if path:
            self.model_folder_edit.setText(path)

    def load_settings(self):
        self.device_combo.setCurrentText(self.settings.value("device", "cuda:0"))
        self.precision_combo.setCurrentText(self.settings.value("precision", "bfloat16"))
        self.model_folder_edit.setText(self.settings.value("model_folder", ""))
        self.torch_compile.setChecked(self.settings.value("torch_compile", False, type=bool))
        self.cpu_offload.setChecked(self.settings.value("cpu_offload", False, type=bool))
        self.overlapped.setChecked(self.settings.value("overlapped", False, type=bool))

    def save_settings(self):
        self.settings.setValue("device", self.device_combo.currentText())
        self.settings.setValue("precision", self.precision_combo.currentText())
        self.settings.setValue("model_folder", self.model_folder_edit.text().strip())
        self.settings.setValue("torch_compile", self.torch_compile.isChecked())
        self.settings.setValue("cpu_offload", self.cpu_offload.isChecked())
        self.settings.setValue("overlapped", self.overlapped.isChecked())

    def save_and_accept(self):
        self.save_settings()
        self.accept()

    def get_settings(self):
        return {
            'device': self.device_combo.currentText(),
            'precision': self.precision_combo.currentText(),
            'model_folder': self.model_folder_edit.text().strip(),
            'torch_compile': self.torch_compile.isChecked(),
            'cpu_offload': self.cpu_offload.isChecked(),
            'overlapped': self.overlapped.isChecked()
        }

# --- Preset Manager Dialog ---
class PresetManagerDialog(QDialog):
    def __init__(self, parent=None, current_params=None):
        super().__init__(parent)
        self.setWindowTitle("Prompt Preset Manager")
        self.setModal(True)
        self.resize(400, 400)
        self.current_params = current_params or {}
        self.presets_file = "ace_step_presets.json"
        self.presets = self.load_presets()
        
        self.setup_ui()
        self.refresh_list()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.preset_list = QListWidget()
        self.preset_list.itemDoubleClicked.connect(self.load_selected_preset)
        layout.addWidget(self.preset_list)

        save_group = QGroupBox("Save Current Configuration")
        save_layout = QHBoxLayout()
        
        self.preset_name_input = QLineEdit()
        self.preset_name_input.setPlaceholderText("Preset name (e.g., Epic Cinematic Choir)")
        save_layout.addWidget(self.preset_name_input)
        
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.save_preset)
        save_layout.addWidget(save_btn)
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)

        btn_layout = QHBoxLayout()
        load_btn = QPushButton("📂 Load Selected")
        load_btn.clicked.connect(self.load_selected_preset)
        delete_btn = QPushButton("🗑️ Delete Selected")
        delete_btn.clicked.connect(self.delete_preset)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(delete_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def load_presets(self):
        if os.path.exists(self.presets_file):
            try:
                with open(self.presets_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_presets_to_file(self):
        with open(self.presets_file, 'w') as f:
            json.dump(self.presets, f, indent=4)

    def refresh_list(self):
        self.preset_list.clear()
        for name in self.presets.keys():
            self.preset_list.addItem(name)

    def save_preset(self):
        name = self.preset_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a preset name.")
            return
        if name in self.presets:
            reply = QMessageBox.question(self, "Confirm", "Preset exists. Overwrite?")
            if reply != QMessageBox.Yes:
                return
                
        self.presets[name] = self.current_params
        self.save_presets_to_file()
        self.refresh_list()
        self.preset_name_input.clear()
        QMessageBox.information(self, "Success", f"Preset '{name}' saved.")

    def load_selected_preset(self, item=None):
        if not item:
            item = self.preset_list.currentItem()
        if not item:
            return
            
        name = item.text()
        if name in self.presets:
            self.selected_preset = self.presets[name]
            self.accept()

    def delete_preset(self):
        item = self.preset_list.currentItem()
        if not item:
            return
        name = item.text()
        reply = QMessageBox.question(self, "Confirm", f"Delete preset '{name}'?")
        if reply == QMessageBox.Yes:
            del self.presets[name]
            self.save_presets_to_file()
            self.refresh_list()


# ==============================================================
# 6. MAIN APPLICATION WINDOW
# ==============================================================
class ACEStepApp(QMainWindow):
    log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.bridge = ACEStepBridge()
        self.current_audio = None
        self.sample_rate = 44100
        self.last_params = {}
        self.settings = QSettings("ACEStep", "ACEStepApp")
        self.history_items = []
        self.uploaded_audio_path = None
        
        # Redirect stdout/stderr to UI
        sys.stdout = LogRedirector(self.log_signal)
        sys.stderr = LogRedirector(self.log_signal)
        
        self.init_ui()
        self.apply_dark_theme()
        self.log_signal.connect(self.append_log)

    # ----------------------------------------------------------
    # 6.1 THEMING & INITIALIZATION
    # ----------------------------------------------------------
    def apply_dark_theme(self):
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(45, 45, 45))
        palette.setColor(QPalette.AlternateBase, QColor(60, 60, 60))
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(50, 50, 50))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.Highlight, QColor(76, 175, 80))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)

    def init_ui(self):
        self.setWindowTitle("ACE-Step Music Studio")
        self.setGeometry(100, 100, 1200, 850)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left Side Setup
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.setup_top_bar(left_layout)
        
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_text2music_tab(), "Text2Music")
        self.tabs.addTab(self.create_retake_tab(), "Retake (Variations)")
        self.tabs.addTab(self.create_repaint_tab(), "Repaint")
        self.tabs.addTab(self.create_edit_tab(), "Lyric Edit")
        left_layout.addWidget(self.tabs, 1)
        
        main_splitter.addWidget(left_widget)
        
        # Right Side Setup
        main_splitter.addWidget(self.setup_history_panel())
        main_splitter.setSizes([800, 300])
        main_layout.addWidget(main_splitter, 1)
        
        # Bottom Area Setup
        bottom_splitter = QSplitter(Qt.Vertical)
        bottom_splitter.addWidget(self.setup_audio_player())
        bottom_splitter.addWidget(self.setup_log_console())
        bottom_splitter.setSizes([200, 100])
        main_layout.addWidget(bottom_splitter)

        # Media Player Setup
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)
        self.media_player.playingChanged.connect(self.update_play_button)
        self.media_player.positionChanged.connect(self.update_position)
        self.media_player.durationChanged.connect(self.update_duration)

    def setup_top_bar(self, parent_layout):
        top_bar = QHBoxLayout()
        self.settings_btn = QPushButton("⚙ Settings & Load Model")
        self.settings_btn.clicked.connect(self.open_settings)
        top_bar.addWidget(self.settings_btn)
        
        self.model_status = QLabel("Model: Not Loaded")
        self.model_status.setStyleSheet("color: red; font-weight: bold;")
        top_bar.addWidget(self.model_status)
        top_bar.addStretch()
        
        current_folder = self.settings.value("model_folder", "", type=str)
        folder_display = current_folder if current_folder else "Default (~/.cache/ace-step)"
        self.folder_label = QLabel(f"📂 {folder_display}")
        self.folder_label.setStyleSheet("color: #888888; font-style: italic;")
        top_bar.addWidget(self.folder_label)
        parent_layout.addLayout(top_bar)

    def setup_history_panel(self):
        right_widget = QGroupBox("Generation History")
        right_layout = QVBoxLayout(right_widget)
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.load_history_item)
        right_layout.addWidget(self.history_list)
        clear_history_btn = QPushButton("🗑️ Clear History")
        clear_history_btn.clicked.connect(self.clear_history)
        right_layout.addWidget(clear_history_btn)
        return right_widget

    def setup_audio_player(self):
        player_group = QGroupBox("Audio Player & Output")
        player_layout = QVBoxLayout()
        
        audio_controls = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        self.play_btn.setShortcut(QKeySequence("Space"))
        audio_controls.addWidget(self.play_btn)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFixedWidth(100)
        audio_controls.addWidget(self.time_label)
        
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setEnabled(False)
        self.seek_slider.sliderMoved.connect(self.seek_audio)
        audio_controls.addWidget(self.seek_slider, 1)
        
        audio_controls.addWidget(QLabel("🔊"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.setFixedWidth(80)
        self.volume_slider.valueChanged.connect(self.change_volume)
        audio_controls.addWidget(self.volume_slider)
        
        self.export_btn = QPushButton("💾 Export")
        self.export_btn.clicked.connect(self.export_audio)
        self.export_btn.setEnabled(False)
        audio_controls.addWidget(self.export_btn)
        player_layout.addLayout(audio_controls)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        player_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        player_layout.addWidget(self.status_label)

        player_group.setLayout(player_layout)
        return player_group

    def setup_log_console(self):
        log_group = QGroupBox("Backend Log Console")
        log_layout = QVBoxLayout(log_group)
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas, monospace;")
        log_layout.addWidget(self.log_console)
        return log_group

    # ----------------------------------------------------------
    # 6.2 TAB CONSTRUCTORS
    # ----------------------------------------------------------
    def create_text2music_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Config Row
        top_config_layout = QHBoxLayout()
        top_config_layout.addWidget(QLabel("Language:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English", "Chinese", "Russian", "Spanish", "Japanese", "German", "French", "Portuguese", "Italian", "Korean"])
        top_config_layout.addWidget(self.lang_combo)
        
        top_config_layout.addWidget(QLabel("Application Mode:"))
        self.app_mode_combo = QComboBox()
        self.app_mode_combo.addItems(["Default", "Lyric2Vocal (LoRA)", "Text2Samples (LoRA)"])
        self.app_mode_combo.setToolTip("Lyric2Vocal: Generates pure vocals from lyrics.\nText2Samples: Generates pure instrumental samples.")
        top_config_layout.addWidget(self.app_mode_combo)
        top_config_layout.addStretch()
        
        preset_btn = QPushButton("⭐ Presets")
        preset_btn.clicked.connect(self.open_preset_manager)
        top_config_layout.addWidget(preset_btn)
        layout.addLayout(top_config_layout)
        
        # Tags
        layout.addWidget(QLabel("Tags (Genres, Mood, Instruments, Scenarios):"))
        self.t2m_tags = QLineEdit()
        self.t2m_tags.setPlaceholderText("e.g., pop, rock, emotional, acoustic guitar, cinematic background")
        layout.addWidget(self.t2m_tags)

        chips_layout = QHBoxLayout()
        chips_layout.addWidget(QLabel("Quick Tags:"))
        for tag_text in ["Pop", "Rock", "Cinematic", "Lo-Fi", "EDM", "Acoustic", "Jazz"]:
            chip = QPushButton(tag_text)
            chip.setStyleSheet("padding: 2px 8px; font-size: 10px;")
            chip.clicked.connect(lambda checked, t=tag_text: self.add_quick_tag(t))
            chips_layout.addWidget(chip)
        chips_layout.addStretch()
        layout.addLayout(chips_layout)

        # Vocal/Instrumental Config
        vocal_inst_layout = QHBoxLayout()
        self.instrumental_check = QCheckBox("Instrumental Only (No Vocals)")
        self.instrumental_check.toggled.connect(self.toggle_instrumental)
        vocal_inst_layout.addWidget(self.instrumental_check)
        
        vocal_inst_layout.addWidget(QLabel("Vocal Style:"))
        self.vocal_style_edit = QLineEdit()
        self.vocal_style_edit.setPlaceholderText("e.g., male tenor, female soprano, raspy, choir")
        vocal_inst_layout.addWidget(self.vocal_style_edit)
        layout.addLayout(vocal_inst_layout)

        # Lyrics
        lyrics_header_layout = QHBoxLayout()
        lyrics_header_layout.addWidget(QLabel("Lyrics:"))
        lyrics_header_layout.addStretch()
        for tag in ["[verse]", "[chorus]", "[bridge]", "[instrumental]"]:
            btn = QPushButton(tag)
            btn.setStyleSheet("padding: 2px 5px; font-size: 10px;")
            btn.clicked.connect(lambda checked, t=tag: self.insert_lyric_tag(t))
            lyrics_header_layout.addWidget(btn)
        layout.addLayout(lyrics_header_layout)
        
        self.t2m_lyrics = QTextEdit()
        self.t2m_lyrics.setPlaceholderText("[verse]\nYour lyrics here\n\n[chorus]\nYour chorus here")
        layout.addWidget(self.t2m_lyrics)

        # Params
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Duration (s):"))
        self.t2m_duration = QSpinBox()
        self.t2m_duration.setRange(-1, 300)
        self.t2m_duration.setValue(30)
        params_layout.addWidget(self.t2m_duration)

        params_layout.addWidget(QLabel("Steps:"))
        self.t2m_steps = QSpinBox()
        self.t2m_steps.setRange(10, 100)
        self.t2m_steps.setValue(27)
        params_layout.addWidget(self.t2m_steps)

        params_layout.addWidget(QLabel("Guidance:"))
        self.t2m_guidance = QDoubleSpinBox()
        self.t2m_guidance.setRange(1.0, 10.0)
        self.t2m_guidance.setValue(3.0)
        params_layout.addWidget(self.t2m_guidance)

        params_layout.addWidget(QLabel("Seed:"))
        self.t2m_seed = QSpinBox()
        self.t2m_seed.setRange(0, 999999)
        self.t2m_seed.setValue(42)
        params_layout.addWidget(self.t2m_seed)

        params_layout.addWidget(QLabel("Batch:"))
        self.batch_count_spin = QSpinBox()
        self.batch_count_spin.setRange(1, 10)
        self.batch_count_spin.setValue(1)
        self.batch_count_spin.setToolTip("Generate multiple variations with incrementing seeds.")
        params_layout.addWidget(self.batch_count_spin)
        layout.addLayout(params_layout)

        gen_btn = QPushButton("🚀 Generate Music")
        gen_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; padding: 10px;")
        gen_btn.clicked.connect(lambda: self.start_generation("text2music"))
        layout.addWidget(gen_btn)
        
        layout.addStretch()
        return tab

    def create_retake_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Retake: Generate variations using flow-matching noise mixing."))
        
        ratio_layout = QHBoxLayout()
        ratio_layout.addWidget(QLabel("Noise Mixing Ratio:"))
        self.variance_slider = QSlider(Qt.Horizontal)
        self.variance_slider.setRange(0, 100)
        self.variance_slider.setValue(30)
        self.variance_label = QLabel("0.30")
        self.variance_slider.valueChanged.connect(lambda v: self.variance_label.setText(f"{v/100.0:.2f}"))
        ratio_layout.addWidget(self.variance_slider)
        ratio_layout.addWidget(self.variance_label)
        layout.addLayout(ratio_layout)
        
        src_layout = QHBoxLayout()
        src_layout.addWidget(QLabel("Source Audio:"))
        self.retake_source_combo = QComboBox()
        self.retake_source_combo.addItems(["Use Last Generated", "Upload Audio"])
        self.retake_source_combo.currentIndexChanged.connect(self.toggle_retake_upload)
        src_layout.addWidget(self.retake_source_combo)
        
        self.retake_upload_btn = QPushButton("📁 Upload Audio")
        self.retake_upload_btn.setEnabled(False)
        self.retake_upload_btn.clicked.connect(lambda: self.upload_source_audio("retake"))
        src_layout.addWidget(self.retake_upload_btn)
        layout.addLayout(src_layout)
        
        btn = QPushButton("🎲 Generate Variation")
        btn.clicked.connect(lambda: self.start_generation("retake"))
        layout.addWidget(btn)
        layout.addStretch()
        return tab

    def create_repaint_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Repaint: Selectively regenerate specific sections (Inpainting)."))
        
        src_layout = QHBoxLayout()
        src_layout.addWidget(QLabel("Source Audio:"))
        self.repaint_source_combo = QComboBox()
        self.repaint_source_combo.addItems(["Use Last Generated", "Upload Audio"])
        self.repaint_source_combo.currentIndexChanged.connect(self.toggle_repaint_upload)
        src_layout.addWidget(self.repaint_source_combo)
        
        self.repaint_upload_btn = QPushButton("📁 Upload Audio")
        self.repaint_upload_btn.setEnabled(False)
        self.repaint_upload_btn.clicked.connect(lambda: self.upload_source_audio("repaint"))
        src_layout.addWidget(self.repaint_upload_btn)
        layout.addLayout(src_layout)
        
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Start Time (s):"))
        self.repaint_start = QSpinBox()
        self.repaint_start.setRange(0, 300)
        time_layout.addWidget(self.repaint_start)
        
        time_layout.addWidget(QLabel("End Time (s):"))
        self.repaint_end = QSpinBox()
        self.repaint_end.setRange(0, 300)
        self.repaint_end.setValue(10)
        time_layout.addWidget(self.repaint_end)
        layout.addLayout(time_layout)
        
        layout.addWidget(QLabel("New Tags/Lyrics for selected section (optional):"))
        self.repaint_tags_edit = QLineEdit()
        self.repaint_tags_edit.setPlaceholderText("Leave blank to preserve original style")
        layout.addWidget(self.repaint_tags_edit)
        
        btn = QPushButton("🖌️ Repaint Section")
        btn.clicked.connect(lambda: self.start_generation("repaint"))
        layout.addWidget(btn)
        layout.addStretch()
        return tab

    def create_edit_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Lyric Editing: Modify lyrics while preserving melody/accompaniment."))
        
        src_layout = QHBoxLayout()
        src_layout.addWidget(QLabel("Source Audio:"))
        self.edit_source_combo = QComboBox()
        self.edit_source_combo.addItems(["Use Last Generated", "Upload Audio"])
        self.edit_source_combo.currentIndexChanged.connect(self.toggle_edit_upload)
        src_layout.addWidget(self.edit_source_combo)
        
        self.edit_upload_btn = QPushButton("📁 Upload Audio")
        self.edit_upload_btn.setEnabled(False)
        self.edit_upload_btn.clicked.connect(lambda: self.upload_source_audio("edit"))
        src_layout.addWidget(self.edit_upload_btn)
        layout.addLayout(src_layout)
        
        layout.addWidget(QLabel("Edit Mode:"))
        mode_layout = QHBoxLayout()
        self.edit_mode_group = QButtonGroup()
        
        radio1 = QRadioButton("Only Lyrics (Preserve Melody)")
        radio1.setChecked(True)
        radio2 = QRadioButton("Remix (Change Melody)")
        
        self.edit_mode_group.addButton(radio1, 1)
        self.edit_mode_group.addButton(radio2, 2)
        
        mode_layout.addWidget(radio1)
        mode_layout.addWidget(radio2)
        layout.addLayout(mode_layout)
        
        layout.addWidget(QLabel("New Lyrics:"))
        self.edit_lyrics_edit = QTextEdit()
        self.edit_lyrics_edit.setPlaceholderText("Paste the full lyrics here, with the modifications you want.")
        layout.addWidget(self.edit_lyrics_edit)
        
        btn = QPushButton("✏️ Apply Edit")
        btn.clicked.connect(lambda: self.start_generation("edit"))
        layout.addWidget(btn)
        layout.addStretch()
        return tab

    # ----------------------------------------------------------
    # 6.3 UI HELPER METHODS
    # ----------------------------------------------------------
    def insert_lyric_tag(self, tag):
        cursor = self.t2m_lyrics.textCursor()
        cursor.insertText(f"{tag}\n")
        self.t2m_lyrics.setTextCursor(cursor)
        self.t2m_lyrics.setFocus()

    def open_preset_manager(self):
        current_params = self.get_text2music_params()
        dialog = PresetManagerDialog(self, current_params)
        if dialog.exec():
            if hasattr(dialog, 'selected_preset'):
                preset = dialog.selected_preset
                self.t2m_tags.setText(preset.get('tags', ''))
                self.t2m_lyrics.setPlainText(preset.get('lyrics', ''))
                self.lang_combo.setCurrentText(preset.get('language', 'English'))
                self.app_mode_combo.setCurrentText(preset.get('app_mode', 'Default'))
                self.vocal_style_edit.setText(preset.get('vocal_style', ''))
                self.instrumental_check.setChecked(preset.get('instrumental', False))
                self.t2m_duration.setValue(preset.get('duration', 30))
                self.t2m_steps.setValue(preset.get('steps', 27))
                self.t2m_guidance.setValue(preset.get('guidance_scale', 3.0))
                self.t2m_seed.setValue(preset.get('seed', 42))
                QMessageBox.information(self, "Loaded", "Preset loaded successfully.")

    def get_text2music_params(self):
        return {
            'tags': self.t2m_tags.text(),
            'lyrics': self.t2m_lyrics.toPlainText(),
            'language': self.lang_combo.currentText(),
            'app_mode': self.app_mode_combo.currentText(),
            'instrumental': self.instrumental_check.isChecked(),
            'vocal_style': self.vocal_style_edit.text(),
            'duration': self.t2m_duration.value(),
            'steps': self.t2m_steps.value(),
            'guidance_scale': self.t2m_guidance.value(),
            'seed': self.t2m_seed.value()
        }

    def toggle_instrumental(self, checked):
        self.t2m_lyrics.setEnabled(not checked)
        self.vocal_style_edit.setEnabled(not checked)
        if checked:
            self.t2m_lyrics.clear()

    def add_quick_tag(self, tag):
        current_tags = self.t2m_tags.text().strip()
        if not current_tags:
            self.t2m_tags.setText(tag.lower())
        else:
            existing = [t.strip().lower() for t in current_tags.split(',')]
            if tag.lower() not in existing:
                self.t2m_tags.setText(f"{current_tags}, {tag.lower()}")

    def toggle_retake_upload(self, index):
        self.retake_upload_btn.setEnabled(index == 1)

    def toggle_repaint_upload(self, index):
        self.repaint_upload_btn.setEnabled(index == 1)

    def toggle_edit_upload(self, index):
        self.edit_upload_btn.setEnabled(index == 1)

    def upload_source_audio(self, context):
        path, _ = QFileDialog.getOpenFileName(self, "Upload Source Audio", "", "Audio Files (*.wav *.mp3 *.flac)")
        if path:
            self.uploaded_audio_path = path
            self.status_label.setText(f"Uploaded: {os.path.basename(path)}")

    def append_log(self, text):
        self.log_console.append(text)
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    # ----------------------------------------------------------
    # 6.4 SETTINGS & MODEL LOADING
    # ----------------------------------------------------------
    def open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec():
            settings = dialog.get_settings()
            folder_display = settings['model_folder'] if settings['model_folder'] else "Default (~/.cache/ace-step)"
            self.folder_label.setText(f"📂 {folder_display}")
            self.status_label.setText("Loading model...")
            QApplication.processEvents()
            
            try:
                self.bridge.load_model(
                    device=settings['device'],
                    precision=settings['precision'],
                    model_folder=settings['model_folder'],
                    torch_compile=settings['torch_compile'],
                    cpu_offload=settings['cpu_offload'],
                    overlapped_decode=settings['overlapped']
                )
                self.model_status.setText("Model: Loaded")
                self.model_status.setStyleSheet("color: green; font-weight: bold;")
                self.status_label.setText("Model loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")
                self.status_label.setText("Model loading failed")

    # ----------------------------------------------------------
    # 6.5 GENERATION LOGIC & WORKER SLOTS
    # ----------------------------------------------------------
    def start_generation(self, mode):
        if not self.bridge.is_loaded:
            QMessageBox.warning(self, "Warning", "Please load the model in Settings first!")
            return

        params = self.get_text2music_params()
        params['mode'] = mode
        
        if mode == "retake":
            params['variance_ratio'] = self.variance_slider.value() / 100.0
            params['source_mode'] = self.retake_source_combo.currentText()
            if params['source_mode'] == "Upload Audio" and self.uploaded_audio_path:
                params['source_audio_path'] = self.uploaded_audio_path
            elif params['source_mode'] == "Upload Audio" and not self.uploaded_audio_path:
                QMessageBox.warning(self, "Warning", "Please upload source audio first!")
                return
        elif mode == "repaint":
            params['start_time'] = self.repaint_start.value()
            params['end_time'] = self.repaint_end.value()
            params['repaint_tags'] = self.repaint_tags_edit.text()
            params['source_mode'] = self.repaint_source_combo.currentText()
            if params['source_mode'] == "Upload Audio" and self.uploaded_audio_path:
                params['source_audio_path'] = self.uploaded_audio_path
            elif params['source_mode'] == "Upload Audio" and not self.uploaded_audio_path:
                QMessageBox.warning(self, "Warning", "Please upload source audio first!")
                return
        elif mode == "edit":
            params['edit_mode'] = "only_lyrics" if self.edit_mode_group.checkedId() == 1 else "remix"
            params['edit_lyrics'] = self.edit_lyrics_edit.toPlainText()
            params['source_mode'] = self.edit_source_combo.currentText()
            if params['source_mode'] == "Upload Audio" and self.uploaded_audio_path:
                params['source_audio_path'] = self.uploaded_audio_path
            elif params['source_mode'] == "Upload Audio" and not self.uploaded_audio_path:
                QMessageBox.warning(self, "Warning", "Please upload source audio first!")
                return

        self.last_params = params
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Generating ({mode})...")
        
        self.tabs.setEnabled(False)
        self.settings_btn.setEnabled(False)

        batch_count = self.batch_count_spin.value() if mode == "text2music" else 1

        self.worker = GenerationWorker(self.bridge, params, batch_count)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.step_log.connect(self.append_log)
        self.worker.track_finished.connect(self.generation_track_finished)
        self.worker.batch_finished.connect(self.generation_batch_finished)
        self.worker.error.connect(self.generation_error)
        self.worker.start()

    def generation_track_finished(self, audio_data, sample_rate, params):
        self.current_audio = audio_data
        self.sample_rate = sample_rate
        self.status_label.setText("Track generated! Adding to history...")
        
        self.play_btn.setEnabled(True)
        self.seek_slider.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        item_name = f"{params['mode'].upper()}: {params['tags'][:20]}... | Seed: {params['seed']}"
        self.history_items.append({
            'name': item_name,
            'audio': audio_data,
            'sr': sample_rate,
            'params': params
        })
        list_item = QListWidgetItem(item_name)
        self.history_list.addItem(list_item)
        self.history_list.setCurrentItem(list_item)
        
        self.temp_audio_path = "temp_output.wav"
        sf.write(self.temp_audio_path, self.current_audio, self.sample_rate)
        self.media_player.setSource(QUrl.fromLocalFile(self.temp_audio_path))

    def generation_batch_finished(self):
        self.progress_bar.setVisible(False)
        self.tabs.setEnabled(True)
        self.settings_btn.setEnabled(True)
        self.status_label.setText("Batch generation complete!")

    def generation_error(self, msg):
        self.progress_bar.setVisible(False)
        self.tabs.setEnabled(True)
        self.settings_btn.setEnabled(True)
        self.status_label.setText("Generation failed.")
        QMessageBox.critical(self, "Error", msg)

    # ----------------------------------------------------------
    # 6.6 AUDIO PLAYBACK & EXPORT
    # ----------------------------------------------------------
    def toggle_play(self):
        if self.media_player.isPlaying():
            self.media_player.pause()
        else:
            self.media_player.play()

    def update_play_button(self, playing):
        self.play_btn.setText("⏸ Pause" if playing else "▶ Play")

    def update_position(self, position):
        self.seek_slider.setValue(position)
        current_time = QTime(0, 0, 0).addMSecs(position)
        self.time_label.setText(f"{current_time.toString('mm:ss')} / {self.total_time_str}")

    def update_duration(self, duration):
        self.seek_slider.setRange(0, duration)
        total_time = QTime(0, 0, 0).addMSecs(duration)
        self.total_time_str = total_time.toString('mm:ss')
        self.update_position(self.media_player.position())

    def seek_audio(self, position):
        self.media_player.setPosition(position)

    def change_volume(self, value):
        self.audio_output.setVolume(value / 100.0)

    def export_audio(self):
        if self.current_audio is None:
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Export Audio", "", "WAV Files (*.wav)")
        if path:
            try:
                sf.write(path, self.current_audio, self.sample_rate)
                meta_path = path.replace('.wav', '.json')
                with open(meta_path, 'w') as f:
                    json.dump(self.last_params, f, indent=4)
                QMessageBox.information(self, "Success", f"Exported to {path}\nMetadata saved to {meta_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def load_history_item(self, item):
        idx = self.history_list.row(item)
        if idx < len(self.history_items):
            data = self.history_items[idx]
            self.current_audio = data['audio']
            self.sample_rate = data['sr']
            self.last_params = data['params']
            
            sf.write(self.temp_audio_path, self.current_audio, self.sample_rate)
            self.media_player.setSource(QUrl.fromLocalFile(self.temp_audio_path))
            self.status_label.setText(f"Loaded from history: {data['name']}")

    def clear_history(self):
        self.history_list.clear()
        self.history_items.clear()
        self.status_label.setText("History cleared.")


# ==============================================================
# 7. APPLICATION ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ACEStepApp()
    window.show()
    sys.exit(app.exec())