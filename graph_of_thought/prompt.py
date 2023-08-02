determine_subtasks_prompt = """\
Given a task, write a list of subtasks that must be fulfilled to finish the original task.

Example:
```
Task: "<some task>"

Subtasks: ["<some subtask 1>", "<some subtask 1>"]
```

Task: {task}

Subtasks: """  # {subtasks}

determine_subtasks_interdependencies_prefix = """\
Given a task and a list of subtasks, determine the dependency relationships between the subtasks.

Example:
```
Task: "<some task>"

Subtasks:
1. "<subtask 1>"
2. "<subtask 2>"
3. "<subtask 3>"

Question: To fulfill "<subtask 1>", what subtasks must be finished first (2, 3)?
Answer: [2, 3]

Question: To fulfill "<subtask 2>", what subtasks must be finished first (1, 3)?
Answer: [3]

(...)
```
{history}
"""

determine_subtasks_interdependencies_prompt = """\
Question: To fulfill {current_subtask}, what subtasks must be finished first {current_options}?
Answer: """  # {current_dependencies}
