import re
import pandas as pd

# Define the file path
file_path = "eval_results/result_evaluation_v5_starting_5_deterministic_smooth_0.4.txt"

# Define regex patterns to extract relevant information
task_pattern = re.compile(r"Task: (.*?), Room Idx: (\d+), Episode Label: (\d+), Trial Label: (\d+), Result: (True|False)")
success_pattern = re.compile(r"success: ([0-9.]+)")
reach_success_pattern = re.compile(r"reach_success: ([0-9.]+)")
lift_success_pattern = re.compile(r"lift_success: ([0-9.]+)")

# Order for tasks
task_order = [
    "Stack-Single-Cube",
    "Push-Box",
    "Open-Drawer",
    "Close-Drawer",
    "Flip-Mug",
    "Press-Gamepad-Blue",
    "Press-Gamepad-Red",
    "Stack-Single-Cube-From-Drawer",
    "Insert-And-Unload-Cans",
    "Orient-Pour-Balls",
    "Sort-Cans",
    "Press-Gamepad-Blue-Red"
]

# Store data in a list of dictionaries
data = []

# Read and parse the file
with open(file_path, "r") as file:
    current_task = None
    for line in file:
        task_match = task_pattern.match(line)
        if task_match:
            current_task = {
                "Task": task_match.group(1),
                "Room Index": int(task_match.group(2)),
                "Episode Label": int(task_match.group(3)),
                "Trial Label": int(task_match.group(4)),
                "Result": task_match.group(5) == "True",
            }
        elif current_task:
            if match := success_pattern.search(line):
                current_task["Success"] = float(match.group(1))
            if match := reach_success_pattern.search(line):
                current_task["Reach Success"] = float(match.group(1))
            if match := lift_success_pattern.search(line):
                current_task["Lift Success"] = float(match.group(1))
            # if "Success" in current_task and "Reach Success" in current_task and "Lift Success" in current_task:
            data.append(current_task)
            current_task = None

# Create a DataFrame
df = pd.DataFrame(data)

# Aggregate success rates by task
summary = df.groupby("Task").mean().reset_index()

# Order tasks
summary["Order"] = summary["Task"].apply(lambda x: task_order.index(x) if x in task_order else len(task_order))
summary = summary.sort_values(by="Order").drop(columns=["Order"])

# Display the ordered table
print(summary)

# Save to a CSV file
summary.to_csv("ordered_success_rates_summary.csv", index=False)
