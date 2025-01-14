"""
Human-Robot Interaction - simulated
---
Author: Isla Xi Han; Date: 2024/07-2025/01
    Author afficilations:
        Lab for Creative Computation(CRCL), EPFL. url: https://www.crclcrclcrcl.org/
        School of Architecture, Princeton University
    Contact: xihan@princeton.edu
---
Overview:
This program simulates how human can interact with a robot agent through voice control.

Task: 
1. Accepts human input in the form of voice commands (assumes the human is standing and facing the robotic arm for orientation).
2. Translates natural language commands into executable Python code using the OpenAI API.
3. Confirm with human agent if the planned trajactory is correct before execution. 

---
Requirements:
- OpenAI API (version: 1.35.15).
- An OpenAI API Key is required for this program to function.
  * How to create and export an API key: https://platform.openai.com/docs/quickstart
- Windows operating system (otherwise the AudioToText function under HRILLM.py needs to be redefined.)
---
Optional: COMPAS FAB or a similar robotics framework for implementing the generated code on a physical robotic system.
"""
# Import prerequisite libraries
import os
from openai import OpenAI
from playsound import playsound
from HRILLM import AudioToText, generate_response_robmove, TextToAudio, generate_response_ask4conf, confirm2action

# locate audio streaming file location
audioStreamFilePath = "AudioStream\speech.mp3"

# set API key
client = OpenAI(
# defaults to os.environ.get("OPENAI_API_KEY")
api_key=os.getenv("OPENAI_API_KEY"),
)

def main(client, audioStreamFilePath):
    try:
        # convert audio input to text (speech recognition)
        prompt = AudioToText()     
    except ValueError:
        #handle the case where the input is not recognized as valid audio
        print("don't recognize input")
    
    # generate robot movement trajectory and ask for confirmation
    answerA = generate_response_robmove(prompt, client)[0]
    answerB = generate_response_ask4conf(prompt, answerA, client)[0]

    TextToAudio(answerB, audioStreamFilePath, client)
    print(answerB)

    outcome = confirm2action(audioStreamFilePath, client)
    return outcome

if __name__ == "__main__":
    # initialize the flag to track whether the move was successful
    move_successful = False

    # loop until successful execution
    while not move_successful:
        result = main(client, audioStreamFilePath)
        if result == 1: 
            move_successful = True




