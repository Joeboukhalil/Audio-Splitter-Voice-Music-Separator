# Audio Separator

**Author:** Joe Bou Khalil  
**License:** MIT License  

---

## Overview

Audio Separator is a Python-based tool that allows you to:  

1. **Music Separation** – Separate music tracks into different components such as vocals, drums, and guitar.  
2. **Voice Separation** – Isolate human speech from background noise or music, creating a clean voice track.  

This project includes two main functionalities: recording audio from a microphone or uploading audio files, then processing them to extract either music stems or voice-only tracks.

---

## Features

- Record audio from your microphone.
- Upload audio files from your computer.
- Separate music into stems (vocals, drums, guitar, etc.).
- Isolate voice using frequency-based filtering.
- Save the processed audio to your computer.
- Compatible with Python 3.13+ (voice separation) and 3.10+ (music separation with Spleeter).

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Joeboukhalil/Audio-Splitter-Voice-Music-Separator.git
cd audio-separator

2. Install dependencies for voice separation (Python 3.13+):

pip install numpy sounddevice soundfile scipy librosa

3. For music separation (Spleeter, Python 3.10+ recommended):
pip install spleeter tensorflow==2.12.0

## Usage
### Voice Separation (Python 3.13+)
python voice_separator.py

- Enter the duration of the recording when prompted.

- The tool will record your voice, filter it, and save it as voice_only.wav.


### Music Separation (Python 3.13+)
- python audio_separator.py
- Upload your audio file.

The tool will separate it into stems (vocals, drums, guitar, etc.) in the output folder.


## License
This project is licensed under the MIT License. See the LICENSE
 file for details.
