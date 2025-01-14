"""
Human-Robot Interaction with Vocal Control Using Large Language Models
---
Author: Isla Xi Han; Date: 2024/11/21
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

# load testing prompts
file_path = "TextData\Prompts_TestData.xlsx" # Excel file of prompts
df = pd.read_excel(file_path)

client = OpenAI(
# defaults to os.environ.get("OPENAI_API_KEY")
api_key=os.getenv("OPENAI_API_KEY"),
)

def main():
    # extract prompt contents into a list
    prompt_list = df['Prompt Contents'].tolist()
    # prepare a list to restore results
    completion_list = []

    # iterate through prompts
    for prompt in prompt_list:
        response = client.chat.completions.create(
        # Use GPT 3.5 as the LLM
        model="gpt-3.5-turbo",
        temperature=0,
        # Pre-define conversation messages for the possible roles
        messages=[
            {"role": "system", "content": """Interpret the user's input to control the robot's movement in millimeters along the x, y, and z axes. Strictly follow these rules:  
            1) Convert all units to millimeters as follows:  
                - Millimeters (mm): Use the value as-is.  
                - Centimeters (cm): Multiply by 10 to convert to millimeters.  
                - Meters (m): Multiply by 1000 to convert to millimeters.  
                - Inches (in): Multiply by 25.4 to convert to millimeters.  
                - Feet (ft): Multiply by 304.8 to convert to millimeters.  
            2) If no unit is specified, assume the value is in millimeters.  
            3) If user mentions 'a little', 'slightly', 'a bit', 'a little bit' or similar, set the distance to 1 milimeter. 
            4) If no indication of distance is detected, output 'delta_x, delta_y, delta_z = 0.0, 0.0, 0.0' 
            5) Use directional keywords to adjust the axis:  
                - Horizontal (x-axis): If the user mentions 'right', 'move rightward', 'move to the right', 'positive x-axis', 'along x-axis', 'in x direction', 'along x direction' or similar, set delta_x to a positive value (e.g., delta_x = 1). If the user mentions 'left', 'move leftward', 'move to the left', 'negaive x-axis', 'negative x direction' or similar, set delta_x to a negative value (e.g., delta_x = -1).  
                - Horizontal (y-axis): If the user mentions 'back', 'away', 'positive y-axis', 'along y-axis', 'in y direction', 'along y direction' or similar, set delta_y to a positive value (e.g., delta_y = 1). If the user mentions 'forward', 'forwards', 'move forward', 'approach', 'move toward me', 'negaive y-axis', 'negative y direction' or similar, set delta_y to a negative value (e.g., delta_y = -1).  
                - Vertical (z-axis): If the user mentions 'up', 'raise', 'ascend', 'upwards', 'upward', 'increase height', 'lift', 'positive z-axis', 'along z-axis', 'in z direction', , 'along z direction' or similar, set delta_z to a positive value (e.g., delta_z = 1). If the user mentions 'down', 'downward', 'downwards', 'lower', 'descend', 'decrease height', 'negaive z-axis', 'negative z direction' or similar, set delta_z to a negative value (e.g., delta_z = -1).  
            6) Ensure the output is always in this exact format:  
                - delta_x, delta_y, delta_z = a, b, c; where a, b, and c are floats representing distances in millimeters.    
            7) If the input mentions multiple directions or units, convert and apply each direction seperately, then output in the correct format.(e.g., 'move 2 cm up and 3 mm to the right', output 'delta_x, delta_y, delta_z = 3.0, 0.0, 20.0').
            8) Never include any additional characters such as semicolons, quotes, or text outside of the output format."""},
            
            # Few-shot examples:           
            {"role": "user", "content": "Move up 36mm"},
            {"role": "assistant", "content": "delta_x, delta_y, delta_z = 0.0, 0.0, 36.0"},

            {"role": "user", "content": "Move backwards a lil bit"},
            {"role": "assistant", "content": "delta_x, delta_y, delta_z = 0.0, 1.0, 0.0"},

            {"role": "user", "content": "Shift left for 54cm"},
            {"role": "assistant", "content": "delta_x, delta_y, delta_z = -540.0, 0.0, 0.0"},

            {"role": "user", "content": "lower down for 70 centimeters and move to the left for 3 millimeters"},
            {"role": "assistant", "content": "delta_x, delta_y, delta_z = -3.0, 0.0, -700.0"},

            {"role": "user", "content": "lift the arm for 33 mm and forward for 13mm"},
            {"role": "assistant", "content": "delta_x, delta_y, delta_z = 0, -13.0, 33.0"},

            {"role": "user", "content": "move closer for 1 ft"},
            {"role": "assistant", "content": "delta_x, delta_y, delta_z = 0.0, -304.8, 0.0"},

            {"role": "user", "content": "Move away for 16 cm"},
            {"role": "assistant", "content": "delta_x, delta_y, delta_z = 0.0, 160.0, 0.0"},

            {"role": "user", "content": "Slide in the negative Y direction for 23cm"},
            {"role": "assistant", "content": "delta_x, delta_y, delta_z = 0.0, -230.0, 0.0"},

            {"role": "user", "content": "Shift along x-axis for 0.5 m"},
            {"role": "assistant", "content": "delta_x, delta_y, delta_z = 500.0, 0.0, 0.0"},

            {"role": "user", "content": "move towards me a little bit, move to the left for 6 cm, and lower for 12 mm"},
            {"role": "assistant", "content": "delta_x, delta_y, delta_z = -60.0, -1.0, -12.0"},

            {"role": "user", "content": "slide to the right for 40 mm, upward for 33 mm, and away for 54 mm"},
            {"role": "assistant", "content": "delta_x, delta_y, delta_z = 40.0, 54.0, 33.0"},

            # Real Question
            {"role": "user", "content": prompt}
        ]
        )
        # Print and return the cleaned response content
        response_content = response.choices[0].message.content
        completion_list.append(response_content)

    # Add the responses as a new column to the DataFrame
    df['Completion'] = completion_list
    # Save the updated DataFrame back to the same Excel file
    df.to_excel(file_path, index=False)
    print(f"Completion added to a new column in {file_path}")

    # [optional] Extract last token usage metadata
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    # Print the information
    print(f"Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}, Total Tokens: {total_tokens}")

    return

if __name__ == "__main__":
    # Start the timer
    start_time = time.time()

    main()

    # Stop the timer
    end_time = time.time()

    # Calculate execution time
    execution_time = end_time - start_time
    # Calculate per prompt time
    promptCount = 100
    execution_time_perP = execution_time/promptCount
    print(f"Total Execution time: {execution_time:.5f} seconds")
    print(f"Execution time per Prompt: {execution_time_perP:.5f} seconds")




