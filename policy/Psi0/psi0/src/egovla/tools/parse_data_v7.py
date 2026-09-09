import re
import pandas as pd

# Define the file path
file_path = "playground_clean_eval/result_log.txt"

# Define regex patterns to extract relevant information
task_pattern = re.compile(
    r"Task: (.*?), Room Idx: (\d+), Table Idx: (\d+), Episode Label: ([\w\.]+), Trial Label: (\d+), Result: (True|False)"
)

success_pattern = re.compile(r"(success[^:]*): ([0-9.]+)")

# Order for tasks
task_order = [
    "Stack-Can",
    "Push-Box",
    "Open-Drawer",
    "Close-Drawer",
    "Flip-Mug",
    "Pour-Balls",
    "Open-Laptop"
]

long_task_order = [
    "Insert-And-Unload-Cans",
    "Stack-Can-Into-Drawer",
    "Sort-Cans",
    "Unload-Cans",
    "Insert-Cans"
]

# Store data in a list of dictionaries
data = []

from collections import defaultdict
task_dict = {}

count_task_dict = defaultdict(int)
success_task_dict = defaultdict(float)
progress_task_dict = defaultdict(float)

# Read and parse the file
with open(file_path, "r") as file:
    current_task = None
    for line in file:
        # print(line)
        task_match = task_pattern.match(line)
        if task_match:
            current_task = {
                "Task": task_match.group(1),
                "Room Index": int(task_match.group(2)),
                "Table Index": int(task_match.group(3)),
                # "Episode Label": int(task_match.group(4)),
                "Episode Label": task_match.group(4),
                "Trial Label": int(task_match.group(5)),
                "Result": task_match.group(5) == "True",
            }
        elif current_task:
            # for match in success_pattern.finditer(line):
            #     key = match.group(1).strip()
            #     value = float(match.group(2))
            items = line.split(" ")
            # # print(items)
            # print(current_task["Table Index"])
            # if current_task["Table Index"] == 1:
            #     continue
            current_task["success"] = float(items[1])
            success = float(items[1])
            # print(items[2:])
            subtask_progress = 0
            total_subtasks = len(items) // 2
            for i in range(len(items) // 2):
                subtask_progress += int(float(items[i * 2 + 1]) > 0) if success < 0.5 else 1
            # if any(key for key in current_task.keys() if "success" in key.lower()):
            # current_task["subtask_progress"] = subtask_progress / total_subtasks
            # data.append(current_task)
            # current_task = None
            count_task_dict[current_task["Task"]] += 1
            success_task_dict[current_task["Task"]] += float(items[1])
            progress_task_dict[current_task["Task"]] += subtask_progress / total_subtasks

# Create a DataFrame
# df = pd.DataFrame(data)

# Filter columns to include only keys with "success" in them
# success_column
latex_str = ""
mean_sr = 0
mean_progress = 0
print(success_task_dict)
print(count_task_dict)
for task in task_order:
    print(task)
    print(success_task_dict[task], count_task_dict[task])
    print("Task: {}, SR: {}, Progress: {}".format(
        task,
        success_task_dict[task] / count_task_dict[task],
        progress_task_dict[task] / count_task_dict[task],
    ))

    latex_str += " & {:.2f} & {:.2f} ".format(
        (success_task_dict[task] / count_task_dict[task]) * 100,
        (progress_task_dict[task] / count_task_dict[task]) * 100,
    )
    mean_sr +=  (success_task_dict[task] / count_task_dict[task]) * 100
    mean_progress += (progress_task_dict[task] / count_task_dict[task]) * 100

latex_str += " & {:.2f} & {:.2f} ".format(
    mean_sr / len(task_order),
    mean_progress / len(task_order)
)

print("Short Horizon Latex text")

print(latex_str)


latex_str = ""
mean_sr = 0
mean_progress = 0
for task in long_task_order:
    if task not in success_task_dict:
        continue
    print(task)
    print("Task: {}, SR: {}, Progress: {}".format(
        task,
        success_task_dict[task] / count_task_dict[task],
        progress_task_dict[task] / count_task_dict[task],
    ))

    latex_str += " & {:.2f} & {:.2f} ".format(
        (success_task_dict[task] / count_task_dict[task]) * 100,
        (progress_task_dict[task] / count_task_dict[task]) * 100,
    )
    mean_sr +=  (success_task_dict[task] / count_task_dict[task]) * 100
    mean_progress += (progress_task_dict[task] / count_task_dict[task]) * 100

latex_str += " & {:.2f} & {:.2f} ".format(
    mean_sr / len(long_task_order),
    mean_progress / len(long_task_order)
)

print("Long Horizon Latex text")

print(latex_str)