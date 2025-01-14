"""
Human-Robot Interaction with Vocal Control Using Large Language Models
---
Author: Isla Xi Han; Date: 2024/11/21
    Author afficilations:
        Lab for Creative Computation(CRCL), EPFL. url: https://www.crclcrclcrcl.org/
        School of Architecture, Princeton University
    Contact: xihan@princeton.edu
---
Collection of reusable functions for the rest of the definitions in this folder.
"""

import os
from openai import OpenAI
import time
# import pyautogui
# from playsound import playsound
import pandas as pd

# extract prompts from the DataFrame
def extract_prompts(df, var):
    """Extracts prompt contents from the DataFrame."""
    return df[var].tolist()

# generate a single response using OpenAI API; instruction & few-shot prompting
def generate_response_robmove(prompt, client):
    """
    Generates a response from the OpenAI API based on the given prompt.
    Assumption: 
        The human operator stands facing the robot. 
    Args:
    - prompt (str): The input prompt for the model.
    - client: The OpenAI API client.
    
    Returns:
    - str: The content of the assistant's response.
    """
    response = client.chat.completions.create(
        # use GPT 3.5 as the LLM
        model="gpt-3.5-turbo",
        temperature=0,
        # pre-define conversation messages for the possible roles
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
            
            # few-shot examples:           
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

            # real Question
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content, response.usage

def generate_response_ask4conf(prompt, prompt_exe, client):
    """
    Generates a response from the OpenAI API based on the given prompt.
    Assumption: 
        The human operator stands facing the robot. 
    Args:
    - prompt (str): The input prompt for the model.
    - client: The OpenAI API client.
    
    Returns:
    - str: The content of the assistant's response.
    """
    response = client.chat.completions.create(
        # use GPT 3.5 as the LLM
        model="gpt-3.5-turbo",
        # temperature=1 # default
        # pre-define conversation messages for the possible roles
        messages=[
      {"role": "system", "content": """You are a helpful assistant. 
      Always start your answer by repeating the prompt by replying 'I heard you said ...'.
      Then, say 'I will ...' following a natural language discription of the prompt_exe, where the unit is always in millimeters.
      Always finish with a question for confirmation."""},


      # few-shot examples:   
      {"role": "user", "content": "prompt: Shift along x-axis for 0.5 m; prompt_exe: delta_x, delta_y, delta_z = 500.0, 0.0, 0.0"},
      {"role": "assistant", "content": "I heard you said shift along x-axis for 0.5 m. I will move along positive x-axis for 500 mm. Does that sound good to you?"},

      {"role": "user", "content": "prompt: move towards me a little bit, move to the left for 6 cm, and lower for 12 mm; prompt_exe: delta_x, delta_y, delta_z = -60.0, -1.0, -12.0"},
      {"role": "assistant", "content": "I heard you said move towards me a little bit, move to the left for 6 cm, and lower for 12 mm. I will move along negative x-axis for 60 mm, negative y-axis for 1 mm, and negative z-axis for 12 mm. Is that OK?"},
      
      {"role": "user", "content": "prompt: "+prompt+"; prompt_exe: "+prompt_exe}
    ]
    )
    return response.choices[0].message.content, response.usage

def process_prompts(prompt_list, client, generate_response_func, prompt_exe_list=None):
    """
    Processes a list of prompts, generating a completion for each.
    
    Args:
    - prompt_list (list of str): List of prompts to process.
    - client: The OpenAI API client.
    
    Returns:
    - list: A list of completions.
    - dict: Token usage statistics.
    """
    completions = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    # if prompt_exe is not provided, default to an empty list
    if prompt_exe_list is None:    
        for prompt in prompt_list:
            response_content, usage = generate_response_func(prompt, client)
            completions.append(response_content)
            
            # accumulate token usage
            total_usage['prompt_tokens'] += usage.prompt_tokens
            total_usage['completion_tokens'] += usage.completion_tokens
            total_usage['total_tokens'] += usage.total_tokens
        
        return completions, total_usage
    else:
        # ensure prompt_list and prompt_exe have the same length
        if len(prompt_list) != len(prompt_exe_list):
            raise ValueError("prompt_list and prompt_exe must have the same length")
        
        # process both lists together
        for prompt, prompt_exe in zip(prompt_list, prompt_exe_list):
            response_content, usage = generate_response_func(prompt, prompt_exe, client)
            completions.append(response_content)
            
            # accumulate token usage
            total_usage['prompt_tokens'] += usage.prompt_tokens
            total_usage['completion_tokens'] += usage.completion_tokens
            total_usage['total_tokens'] += usage.total_tokens 

    return completions, total_usage       

# save result to a new excel
def save_to_excel_with_suffix(df, file_path, suffix="_Completion"):
    """
    Saves the updated DataFrame to an Excel file with a suffix added to the original filename.

    Args:
    - df (DataFrame): The updated DataFrame.
    - file_path (str): Original path of the Excel file.
    - suffix (str): Suffix to append to the original filename before the extension.
    """
    # get original file name
    file_dir, file_name = os.path.split(file_path)
    file_base, file_ext = os.path.splitext(file_name)
    # add suffix
    new_file_name = f"{file_base}{suffix}{file_ext}"
    new_file_path = os.path.join(file_dir, new_file_name)
    # save the updated DataFrame to the new file
    df.to_excel(new_file_path, index=False)
    print(f"Completion saved to: {new_file_path}")

# save result to the original excel 
def save_to_excel(df, file_path):
    """
    Saves the updated DataFrame to an Excel file.
    
    Args:
    - df (DataFrame): The updated DataFrame.
    - file_path (str): Path to save the Excel file.
    """
    df.to_excel(file_path, index=False)
    print(f"Completion added to a new column in {file_path}")

# calculate and print execution time per prompt
def execution_time_per_prompt(start_time, end_time, promptCount):
    # calculate execution time
    execution_time = end_time - start_time
    # calculate per prompt time
    execution_time_perP = execution_time/promptCount
    print(f"Total Execution time: {execution_time:.5f} seconds")
    print(f"Execution time per Prompt: {execution_time_perP:.5f} seconds")