# Get the average number of steps for items where "Success" is true
import numpy as np
import json

with open("/home/saumyas/semnav_workspace/src/hydra/outputs/explore_eqa_gemini_images_True/gemini_images_True.json", "r") as f:
    data = json.load(f)

# Initialize a variable to hold the sum of steps
total_steps = []
total_failures_zero_steps = 0
total_successes_zero_steps = 0

# Iterate over the dictionary
for key, value in data.items():
    # Access the 'steps' inside 'metrics' and add it to the total
    if (value["Success"] == True):
        total_steps.append(value['metrics']['steps'])

    if (value['Success'] == False and value['metrics']['steps'] == 0):
        total_failures_zero_steps += 1
    
    if (value['Success'] == True and value['metrics']['steps'] == 0):
        total_failures_zero_steps += 1

# Calculate the average
average_steps_success_true = np.mean(total_steps)
print('Average steps on success: ' + str(average_steps_success_true))

print('Total failures with zero steps: ' + str(total_failures_zero_steps))

print('Total successes with zero steps: ' + str(total_successes_zero_steps))