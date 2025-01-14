"""
Human-Robot Interaction with Vocal Control Using Large Language Models
---
Author: Isla Xi Han; Date: 2024/07-2025/01
    Author afficilations:
        Lab for Creative Computation(CRCL), EPFL. url: https://www.crclcrclcrcl.org/
        School of Architecture, Princeton University
    Contact: xihan@princeton.edu
---
Overview:
This program enables human-robot interaction through voice commands, leveraging the OpenAI API for natural language processing. 
The primary goal is to interpret human voice commands, convert them into executable code to adjust robotic parameters, such as those used in COMPAS FAB.

Task: 
1. Accepts human input in the form of voice commands (assumes the human is standing and facing the robotic arm for orientation).
2. Translates natural language commands into executable Python code using the OpenAI API.
3. The output code can used directly to modify robotic parameters.

Evaluation: 
- Tests the accuracy of this workflow using 100 example prompts.
- Reports the time taken and monetary cost of the workflow at the end of the evaluation.
---
Requirements:
- OpenAI API (version: 1.35.15).
- An OpenAI API Key is required for this program to function.
  * How to create and export an API key: https://platform.openai.com/docs/quickstart
---
Optional: COMPAS FAB or a similar robotics framework for implementing the generated code on a physical robotic system.
"""
# Import prerequisite libraries
import os
from openai import OpenAI
import time
import pyautogui
from playsound import playsound
import pandas as pd
from HRILLM import AudioToText, generate_response_robmove, TextToAudio, generate_response_ask4conf, confirm2action

audioStreamFilePath = "AudioStream\speech.mp3"

client = OpenAI(
# defaults to os.environ.get("OPENAI_API_KEY")
api_key=os.getenv("OPENAI_API_KEY"),
)

def main(client, audioStreamFilePath):
    try:
        prompt = AudioToText()     
    except ValueError:
        print("don't recognize input")
    
    answerA = generate_response_robmove(prompt, client)[0]
    answerB = generate_response_ask4conf(prompt, answerA, client)[0]

    TextToAudio(answerB, audioStreamFilePath)
    print(answerB)

    outcome = confirm2action(audioStreamFilePath)
    return outcome

if __name__ == "__main__":
    successMove = 0
    while successMove == 0:
        newtry = main(client, audioStreamFilePath)
        if newtry == 1: 
            successMove+= 1




