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
1. Accepts human input in natural language form (assumes the human is standing and facing the robotic arm for orientation).
2. Translates natural language commands into executable Python code using the OpenAI API.
3. The output code can used directly to modify robotic parameters.

Evaluation: 
- Tests the accuracy of this workflow using 130 example prompts.
- Reports the average time and token taken to executate this workflow at the end of the evaluation.
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
import pandas as pd
from HRILLM import extract_prompts, generate_response_robmove, process_prompts, save_to_excel_with_suffix, execution_time_per_prompt

# load DataFrame
file_path = "TestData\Prompts_TestData_4_1.xlsx" # Excel file of prompts

# total number of test prompts
promptCount = 130

# set API key
client = OpenAI(
# defaults to os.environ.get("OPENAI_API_KEY")
api_key=os.getenv("OPENAI_API_KEY"),
)

# main function
def main(file_path, client):
    """
    Main function to process prompts, generate completions, and save results.
    
    Args:
    - file_path (str): Path to the Excel file containing the DataFrame.
    - client: The OpenAI API client.
    """
    # load the Excel file into a DataFrame
    df = pd.read_excel(file_path)
    
    # extract prompts
    prompt_list = extract_prompts(df, 'Prompt Contents')
    
    # generate completions
    completions, token_usage = process_prompts(prompt_list, client, generate_response_robmove)
    
    # add completions to the DataFrame
    df['Completion'] = completions
    
    # save the updated DataFrame
    save_to_excel_with_suffix(df, file_path, suffix="_Completion")
    
    # calculate and print token usage statistics per prompt
    token_usage = {key: value / promptCount for key, value in token_usage.items()}
    print(f"Average Token Usage per Prompt: {token_usage}")

if __name__ == "__main__":
    # start the timer
    start_time = time.time()

    # main()
    main(file_path, client)

    # stop the timer
    end_time = time.time()

    # print average execution time per prompt
    execution_time_per_prompt(start_time, end_time, promptCount)




