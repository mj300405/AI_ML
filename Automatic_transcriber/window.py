import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget,
                               QFileDialog, QLabel, QHBoxLayout, QGroupBox)
from PySide6.QtGui import QIcon, QFont
from PySide6.QtCore import QSize

from audio_processor import PianoAudioProcessor

class SoundToNotesApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Sound to Notes Transcription')
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon('icon_path_here.png'))  # Set the window icon

        self.initUI()

        self.pianoAudioProcessor = PianoAudioProcessor()

    def initUI(self):
        # Central widget
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)

        # Main layout
        mainLayout = QVBoxLayout(centralWidget)

        # File upload section
        uploadGroupBox = QGroupBox("Upload Audio")
        uploadLayout = QHBoxLayout()
        self.uploadButton = QPushButton('Upload')
        self.uploadButton.setIcon(QIcon('upload_icon_path_here.png'))  # Add an icon to the button
        self.uploadButton.clicked.connect(self.uploadFile)
        uploadLayout.addWidget(self.uploadButton)
        uploadGroupBox.setLayout(uploadLayout)
        mainLayout.addWidget(uploadGroupBox)

        # Transcription display
        transcriptionGroupBox = QGroupBox("Transcription")
        transcriptionLayout = QVBoxLayout()
        self.transcriptionDisplay = QTextEdit()
        self.transcriptionDisplay.setPlaceholderText('Transcribed notes will be displayed here...')
        transcriptionLayout.addWidget(self.transcriptionDisplay)
        transcriptionGroupBox.setLayout(transcriptionLayout)
        mainLayout.addWidget(transcriptionGroupBox)

        # Controls section
        controlsGroupBox = QGroupBox("Controls")
        controlsLayout = QHBoxLayout()
        self.playButton = QPushButton('Play')
        self.playButton.setIcon(QIcon('play_icon_path_here.png'))  # Add play icon
        self.playButton.clicked.connect(self.playAudio)
        controlsLayout.addWidget(self.playButton)
        
        self.saveButton = QPushButton('Save')
        self.saveButton.setIcon(QIcon('save_icon_path_here.png'))  # Add save icon
        self.saveButton.clicked.connect(self.saveTranscription)
        controlsLayout.addWidget(self.saveButton)
        controlsGroupBox.setLayout(controlsLayout)
        mainLayout.addWidget(controlsGroupBox)

        # Apply stylesheet for styling
        self.applyStylesheet()

    def applyStylesheet(self):
        self.setStyleSheet("""
            QGroupBox {
                font: bold;
                border: 1px solid silver;
                border-radius: 6px;
                margin-top: 20px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                left: 7px;
                padding: 5px 10px 5px 10px;
            }
            QPushButton {
                background-color: #A3C1DA;
                border-style: outset;
                border-width: 2px;
                border-radius: 10px;
                border-color: beige;
                font: bold 12px;
                min-width: 10em;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #6698FF;
                border-style: inset;
            }
            QTextEdit {
                font: 12px;
                border: 1px solid #A3C1DA;
            }
        """)

    def uploadFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.mp3 *.wav)")
        if fileName:
            print(f"File selected: {fileName}")

    def playAudio(self):
        print("Playing audio...")

    def saveTranscription(self):
        print("Saving transcription...")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = SoundToNotesApp()
    mainWindow.show()
    sys.exit(app.exec())
