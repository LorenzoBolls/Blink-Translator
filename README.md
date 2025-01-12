# Blink Translator App

## Overview
The Blink Translator App is a **Computer Vision application** designed to assist individuals with paralysis by translating **blinks** into **Morse code** and then into **alphanumeric characters**. Using real-time video from a webcam, the app detects eye blinks, calculates their duration, and converts them into meaningful text. This innovative tool empowers users to communicate effectively, offering a new level of independence.

---

## Features
- **Blink-to-Morse Code Translation**: Converts detected blinks into Morse code using precise timing thresholds.
- **Morse-to-Alphanumeric Conversion**: Decodes Morse code into readable letters and words for effective communication.
- **Real-Time Graphical Feedback**: Displays blink activity visually to reduce user errors and enhance confidence.
- **FaceMesh Integration**: Tracks and analyzes facial landmarks to detect eye blinks with high accuracy.
- **Text-to-Speech (TTS)**: Converts decoded text into speech for auditory feedback.

---

## Key Technologies
- **OpenCV**: For video processing and facial feature detection.
- **cvzone**: Enhances visual feedback and simplifies interface design.
- **Python**: Core programming language for implementing logic and algorithms.
- **FaceMesh Detector**: Tracks facial landmarks to identify eye movements.
- **pyttsx3**: Provides offline text-to-speech functionality.

---

## How It Works
1. **Video Feed**: Captures real-time video from a webcam.
2. **Eye Tracking**: Uses FaceMesh to identify key facial landmarks around the eyes.
3. **Blink Detection**: Classifies blink durations into:
   - **Short Blink**: Represents a dot (`.`).
   - **Long Blink**: Represents a dash (`-`).
   - **Very Long Blink**: Indicates a word break.
4. **Morse Code Conversion**: Blinks are translated into Morse code and decoded into letters using a predefined dictionary.
5. **Text-to-Speech Feedback**: Decoded letters and complete words are announced audibly using TTS.

---

## Setup and Installation

### Prerequisites
- Python 3.x
- Webcam-enabled device

### Dependencies
Install the required Python libraries:
```bash
pip install opencv-python cvzone pyttsx3
```

### Running the Application
1. Clone the repository:
   ```bash
   git clone https://github.com/LorenzoBolls/Blink-Translator.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Blink-Translator
   ```
3. Run the script:
   ```bash
   python3 prototype_blink_translator.py
   ```
4. Ensure your webcam is enabled. The application will start detecting your blinks.

---

## Usage
1. **Start the Application**:
   - Launch the app and position yourself in front of the webcam.
2. **Blink Patterns**:
   - Use short and long blinks to input Morse code:
     - **Short Blink**: Dot (`.`)
     - **Long Blink**: Dash (`-`)
     - **Very Long Blink**: Word break
3. **Graphical Feedback**:
   - The app displays real-time blink activity on the screen to guide your inputs.
4. **Listen to Output**:
   - The application audibly announces each decoded letter and the final word.

---

## Examples
- **Morse Code Input**: `.- ..-` (short, long, short)
  - Decoded Output: `AU`
- **Word Input**: `. -.-. --- -.. .. -. --. / .. ... / ..-. ..- -.`
  - Decoded Output: `CODING IS FUN`

---

## Demo
[![Blink Translator Demo](https://i.imgur.com/r6WR4db.png)](https://www.youtube.com/watch?v=vT8nqBtlH10)

Click the thumbnail above or watch the full video on YouTube: [Blink Translator Demo](https://www.youtube.com/watch?v=vT8nqBtlH10).


---

## Future Enhancements
- **Customizable Blink Thresholds**: Allow users to personalize blink duration settings.
- **Multilingual Support**: Expand Morse code support to include other alphabets and symbols.

---

## Contributions
Contributions are welcome! Feel free to submit issues or pull requests to improve the app. 

### Guidelines
- Fork the repository.
- Make your changes in a new branch.
- Submit a pull request with a detailed explanation of your changes.

---

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

---

## Acknowledgments
- **OpenCV and cvzone**: For providing the core tools for video processing and facial feature detection.
- **pyttsx3**: For enabling offline text-to-speech functionality.
- **Morse Code Community**: For preserving the legacy of Morse code as a means of communication.
- **Testers and Contributors**: For their invaluable feedback and support in improving the application.