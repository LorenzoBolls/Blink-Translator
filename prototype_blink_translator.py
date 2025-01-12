import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import time
import pyttsx3

# Initialize video capture and FaceMesh detector
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)

# Initialize the TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech
tts_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

# Define the Morse code dictionary
morse_code_dict = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
    '9': '----.', '0': '-----', '.': '.-.-.-', ',': '--..--', '?': '..--..',
    "'": '.----.', '!': '-.-.--', '/': '-..-.', '(': '-.--.', ')': '-.--.-',
    '&': '.-...', ':': '---...', ';': '-.-.-.', '=': '-...-', '+': '.-.-.',
    '-': '-....-', '_': '..--.-', '"': '.-..-.', '@': '.--.-', ' ': '/'
}

# Initialize variables
idList = [159, 23, 130, 243, 386, 253, 362, 359]  # Both eyes landmarks
ratioList = []
blinkCounter = 0
color = (255, 0, 255)
blink_start_time = None
dot_threshold = 350  # Short blink threshold in milliseconds (dot)
dash_threshold = 650  # Long blink threshold in milliseconds (dash)
start_stop_threshold = 3000  # 3000 milliseconds for starting/stopping a word
letter_gap_threshold = 2000  # Time in ms to consider end of a letter
blinks = []
last_blink_time = None  # To track the time between blinks
eyes_closed = False  # Flag to check if the eyes are closed
is_building_word = False  # Flag to indicate if we are building a word
current_word = []  # List to store letters for the current word


def record_blink(start_time):
    global blink_duration, blinks, is_building_word, current_word
    blink_duration = (time.time() - start_time) * 1000  # Duration in milliseconds
    print(f"Blink Duration: {blink_duration} ms")  # Debugging information

    if blink_duration >= start_stop_threshold:  # Long blink (3000 ms) to start/stop a word
        if is_building_word:
            # End word building, decode word and reset
            final_word = ''.join(current_word)
            print(f"Final Word: {final_word}")
            current_word = []  # Clear the word list for the next word

            # Speak the final word
            tts_engine.say(final_word)
            tts_engine.runAndWait()
        else:
            print("Starting a new word...")
        is_building_word = not is_building_word  # Toggle word-building mode

    elif is_building_word:  # Only record dots/dashes while building a word
        if blink_duration <= dot_threshold:  # Short blink (dot)
            blinks.append('.')
            print("Recorded a dot (.)")
        elif blink_duration >= dash_threshold:  # Long blink (dash)
            blinks.append('-')
            print("Recorded a dash (-)")
    
    blink_start_time = None


def morse_to_text(morse_code):
    inverted_dict = {value: key for key, value in morse_code_dict.items()}
    decoded_message = ''
    for code in morse_code.split():
        decoded_message += inverted_dict.get(code, '?')
    return decoded_message.strip()


while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    current_time = time.time() * 1000  # Get current time in milliseconds

    if faces:
        face = faces[0]

        # Landmarks for the left eye
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]

        # Landmarks for the right eye
        rightUp = face[386]
        rightDown = face[253]
        rightLeft = face[362]
        rightRight = face[359]

        # Calculate vertical and horizontal distances for the left eye
        leftLenghtVer, _ = detector.findDistance(leftUp, leftDown)
        leftLenghtHor, _ = detector.findDistance(leftLeft, leftRight)

        # Calculate vertical and horizontal distances for the right eye
        rightLenghtVer, _ = detector.findDistance(rightUp, rightDown)
        rightLenghtHor, _ = detector.findDistance(rightLeft, rightRight)

        # Draw lines for the left eye
        cv2.line(img, leftUp, leftDown, (0, 255, 0), 2)  # Vertical line
        cv2.line(img, leftLeft, leftRight, (0, 255, 0), 2)  # Horizontal line

        # Draw lines for the right eye
        cv2.line(img, rightUp, rightDown, (0, 255, 0), 2)  # Vertical line
        cv2.line(img, rightLeft, rightRight, (0, 255, 0), 2)  # Horizontal line

        # Calculate the blink ratios for both eyes
        leftRatio = leftLenghtVer / leftLenghtHor
        rightRatio = rightLenghtVer / rightLenghtHor

        # Average the blink ratios of both eyes
        blinkRatio = (leftRatio + rightRatio) / 2

        # Convert to percentage and append to ratio list
        ratio = int(blinkRatio * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        # Blink detection logic (same as before)
        if ratioAvg < 35 and not eyes_closed:  # Eyes are now closed
            eyes_closed = True
            blink_start_time = time.time()
            color = (0, 200, 0)
            blinkCounter += 1

        elif ratioAvg >= 35 and eyes_closed:  # Eyes have reopened
            eyes_closed = False
            if blink_start_time is not None:
                record_blink(blink_start_time)
                last_blink_time = current_time
                blink_start_time = None
            color = (255, 0, 255)

        # If enough time has passed since the last blink, decode the letter
        if last_blink_time and (current_time - last_blink_time) > letter_gap_threshold:
            if blinks and is_building_word:
                morse_code = ''.join(blinks)
                decoded_letter = morse_to_text(morse_code)
                print(f"Decoded Letter: {decoded_letter}")
                current_word.append(decoded_letter)  # Add the decoded letter to the current word

                # Speak the decoded letter
                tts_engine.say(decoded_letter)
                tts_engine.runAndWait()

                blinks = []  # Clear the list for the next letter
            last_blink_time = None

        # Draw circles for left and right eye landmarks
        for id in idList:
            cv2.circle(img, face[id], 5, color, cv2.FILLED)

        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100), colorR=color)

        imgPlot = plotY.update(ratioAvg, color)
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Blink Translator", imgStack)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
