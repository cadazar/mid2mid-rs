import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
                             QGraphicsRectItem, QGraphicsLineItem, QFileDialog, QMessageBox)
from PyQt6.QtGui import QPen, QBrush, QColor, QPainter
from PyQt6.QtCore import Qt, QRectF

import midi_toolkit

TICK_SCALE = 0.1
PITCH_HEIGHT = 10

class NoteItem(QGraphicsRectItem):
    def __init__(self, pitch, tick, duration, velocity, channel):
        super().__init__()
        self.pitch = pitch
        self.tick = tick
        self.duration = duration
        self.velocity = velocity
        self.channel = channel
        
        self.setRect(tick * TICK_SCALE, (127 - pitch) * PITCH_HEIGHT, duration * TICK_SCALE, PITCH_HEIGHT)
        self.setBrush(QBrush(QColor("blue")))
        self.setPen(QPen(QColor("darkblue")))
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable)

class PianoRoll(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.score = None
        self.notes = []
        self.ticks_per_quarter = 480
        self.time_sig = (4, 4)
        
        self.scene.setSceneRect(0, 0, 10000, 128 * PITCH_HEIGHT)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        
    def load_score(self, score):
        self.score = score
        self.score.merge_tracks()
        self.scene.clear()
        self.notes = []
        self.ticks_per_quarter = score.ticks_per_quarter
        
        self.time_sig = (4, 4)
        if score.tracks:
            for ev in score.tracks[0].events:
                if ev.event_type == "TimeSignature":
                    self.time_sig = (ev.numerator, ev.denominator)
                    break
        
        self.draw_grid()
        
        if not score.tracks:
            return
            
        active_notes = {}
        for ev in score.tracks[0].events:
            if ev.event_type == "NoteOn" and ev.velocity > 0:
                active_notes[(ev.channel, ev.pitch)] = ev
            elif ev.event_type == "NoteOff" or (ev.event_type == "NoteOn" and ev.velocity == 0):
                key = (ev.channel, ev.pitch)
                if key in active_notes:
                    note_on = active_notes.pop(key)
                    duration = ev.tick - note_on.tick
                    note_item = NoteItem(ev.pitch, note_on.tick, duration, note_on.velocity, ev.channel)
                    self.scene.addItem(note_item)
                    self.notes.append(note_item)
                    
        max_tick = max([n.tick + n.duration for n in self.notes], default=1000)
        self.scene.setSceneRect(0, 0, max_tick * TICK_SCALE + 1000, 128 * PITCH_HEIGHT)
        self.update() # Visual refresh on load

    def draw_grid(self):
        pen_black = QPen(QColor(200, 200, 200))
        pen_white = QPen(QColor(230, 230, 230))
        for p in range(128):
            is_black = p % 12 in [1, 3, 6, 8, 10]
            rect = QGraphicsRectItem(0, (127 - p) * PITCH_HEIGHT, 100000, PITCH_HEIGHT)
            rect.setBrush(QBrush(QColor(220, 220, 220) if is_black else QColor(255, 255, 255)))
            rect.setPen(Qt.PenStyle.NoPen)
            rect.setZValue(-2)
            self.scene.addItem(rect)
            
            line = QGraphicsLineItem(0, (127 - p) * PITCH_HEIGHT, 100000, (127 - p) * PITCH_HEIGHT)
            line.setPen(pen_black if is_black else pen_white)
            line.setZValue(-1)
            self.scene.addItem(line)
            
        ticks_per_beat = self.ticks_per_quarter * 4 / self.time_sig[1]
        ticks_per_measure = ticks_per_beat * self.time_sig[0]
        
        pen_beat = QPen(QColor(200, 200, 200))
        pen_measure = QPen(QColor(100, 100, 100), 2)
        
        max_ticks = 100000
        for tick in range(0, max_ticks, int(ticks_per_beat)):
            x = tick * TICK_SCALE
            line = QGraphicsLineItem(x, 0, x, 128 * PITCH_HEIGHT)
            if tick % int(ticks_per_measure) == 0:
                line.setPen(pen_measure)
            else:
                line.setPen(pen_beat)
            line.setZValue(-1)
            self.scene.addItem(line)
            
    def save_score(self, path):
        if not self.score:
            self.score = midi_toolkit.Score(1, 480)
            self.score.tracks.append(midi_toolkit.Track("Track 1"))
            
        if not self.score.tracks:
            self.score.tracks.append(midi_toolkit.Track("Track 1"))
            
        track = self.score.tracks[0]
        new_events = []
        for ev in track.events:
            if ev.event_type not in ["NoteOn", "NoteOff"]:
                new_events.append(ev)
                
        for item in self.scene.items():
            if isinstance(item, NoteItem):
                total_x = item.rect().x() + item.pos().x()
                total_y = item.rect().y() + item.pos().y()
                
                real_tick = int(total_x / TICK_SCALE)
                real_pitch = 127 - int(total_y / PITCH_HEIGHT)
                
                if real_pitch < 0: real_pitch = 0
                if real_pitch > 127: real_pitch = 127
                if real_tick < 0: real_tick = 0
                
                real_duration = int(item.rect().width() / TICK_SCALE)
                
                note_on = midi_toolkit.Event.note_on(real_tick, item.channel, real_pitch, item.velocity)
                note_off = midi_toolkit.Event.note_off(real_tick + real_duration, item.channel, real_pitch, 0)
                
                new_events.extend([note_on, note_off])
                
        track.events = new_events
        track.sort_events()
        self.score.to_file(path)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            pos = self.mapToScene(event.pos())
            tick = int(pos.x() / TICK_SCALE)
            pitch = 127 - int(pos.y() / PITCH_HEIGHT)
            if 0 <= pitch <= 127:
                note = NoteItem(pitch, tick, self.ticks_per_quarter, 100, 0)
                self.scene.addItem(note)
                
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            for item in self.scene.selectedItems():
                if isinstance(item, NoteItem):
                    self.scene.removeItem(item)
        super().keyPressEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MIDI Editor")
        self.resize(800, 600)
        
        self.piano_roll = PianoRoll()
        self.setCentralWidget(self.piano_roll)
        
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        
        open_action = file_menu.addAction("Open")
        open_action.triggered.connect(self.open_file)
        
        save_action = file_menu.addAction("Save")
        save_action.triggered.connect(self.save_file)

        self.current_file = None

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open MIDI File", "", "MIDI Files (*.mid *.midi)")
        if path:
            try:
                score = midi_toolkit.Score.from_file(path)
                self.piano_roll.load_score(score)
                self.current_file = path
                self.setWindowTitle(f"MIDI Editor - {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open file: {e}")

    def save_file(self):
        if not self.current_file:
            path, _ = QFileDialog.getSaveFileName(self, "Save MIDI File", "", "MIDI Files (*.mid *.midi)")
            if not path:
                return
            self.current_file = path
        
        try:
            self.piano_roll.save_score(self.current_file)
            self.setWindowTitle(f"MIDI Editor - {os.path.basename(self.current_file)}")
            QMessageBox.information(self, "Saved", "File saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
