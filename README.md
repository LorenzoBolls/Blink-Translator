# Blink Translator App

## Overview
The Blink Translator App is a **Computer Vision application** designed to assist individuals with paralysis in translating **blinks** into **Morse code** and subsequently into **alphanumeric letters**. By using real-time video feed from a webcam, the app detects eye blinks, calculates blink durations, and converts them into meaningful text.

This project aims to provide a communication method for paralyzed individuals, empowering them to express themselves effectively.

---

## Features
- **Blink-to-Morse Code Translation**: Converts detected blinks into Morse code using precise timing thresholds.
- **Real-Time Feedback**: Provides graphical feedback to visualize blinks, reducing user error and improving confidence.
- **Morse-to-Alphanumeric Conversion**: Decodes Morse code into readable letters and words for effective communication.
- **FaceMesh Detector Integration**: Utilizes advanced facial landmark detection to track eye movements with high accuracy.

---

## Key Technologies
- **OpenCV**: For video processing and facial feature detection.
- **cvzone**: To enhance visual feedback and interface usability.
- **Python**: Core programming language for implementing logic and algorithms.
- **FaceMesh Detector**: Tracks and analyzes facial landmarks to detect eye blinks.

---

## How It Works
1. **Video Feed**: The application captures real-time video from a webcam.
2. **Eye Tracking**: The FaceMesh detector identifies key facial landmarks around the eyes.
3. **Blink Detection**: Blink durations are measured and categorized into:
   - **Dots (.)** for short blinks.
   - **Dashes (-)** for longer blinks.
   - **Word breaks** for very long blinks.
4. **Morse Code Conversion**:
   - Blinks are translated into Morse code sequences.
   - Morse code is converted into alphanumeric characters using a predefined dictionary.
5. **Graphical Feedback**:
   - Blink activity is displayed in real-time to ensure accuracy and reduce errors.

---

## Setup and Installation
### Prerequisites
- Python 3.x
- Webcam-enabled device

### Dependencies
Install the required Python libraries:
```bash
pip install opencv-python cvzone
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
   - The app displays real-time blink activity to guide your inputs.
4. **Decoding**:
   - The app decodes Morse code into alphanumeric letters and words, displaying the results in the console.

---

## Examples
- **Morse Code Input**: `.- ..-` (short, long, short)
  - Decoded Output: `AU`
- **Word Input**: `. -.-. --- -.. .. -. --. / .. ... / ..-. ..- -.`
  - Decoded Output: `CODING IS FUN`

---

## Future Enhancements
- **Text-to-Speech Integration**: Convert decoded text into speech for auditory feedback.
- **Customizable Blink Thresholds**: Allow users to personalize blink duration settings.
- **Support for Additional Languages**: Expand Morse code support to include other alphabets and symbols.

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
- **OpenCV and cvzone**: For providing the foundational tools for real-time video processing.
- **The Morse Code Community**: For keeping this historic communication method alive.
- **Contributors and Testers**: For their invaluable feedback in improving the application.

