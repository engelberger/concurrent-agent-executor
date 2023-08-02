# Graph of Thought

1. Task
2. Determining subtasks
3. Determining subtasks' interdependencies
4. Parallel toposort over subtasks
5. Execute next available layer of subtasks
6. Wait for results (agent could receive extra input)
7. Re-evaluate subtasks' interdependencies; could introduce new tasks
8. If not finished, (4); else (9)
9. Output

## 1. Input

## 2. Determining subtasks

```plain
Given a task, write a list of subtasks that must be fulfilled to finish the original task.

Example:
\`\`\`
Task: "<some task>"

Subtasks: ["<some subtask 1>", "<some subtask 1>"]
\`\`\`

Task: {task}

Subtasks: {subtasks}
```

input: `task: str`
output: `subtasks: list[str]`

## 3. Determining subtasks' interdependencies

prefix:

```plain
Given a task and a list of subtasks, determine the dependency relationships between the subtasks.

Example:
\`\`\`
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
\`\`\`
```

prompt:

```plain
Question: To fulfill {current_subtask}, what subtasks must be finished first {current_options}?
Answer: {current_dependencies}
```

input: `task: str, subtasks: list[str]`
output: `adjacency_lists: list[list[int]]`

## 4. Parallel toposort over subtasks

input: `subtasks_finished: list[bool], adjacency_lists: list[list[int]]`
output: `next_available_layer: list[int]`

## 5. Execute next available layer of subtasks

input: `subtasks: list[str], subtasks_results: list[str], next_available_layer: list[int], tools: list[Tool]`
output: `jobs: list[Promise]`

## 6. Wait for results (agent could receive extra input)

This should consider how to pass the conversational context to each of the steps. Once all the running tasks are finished, should update `subtasks_finished` and `subtasks_results`.

## 7. Re-evaluate subtasks' interdependencies; could introduce new tasks

prefix:

```plain
Given a task, a list of subtasks and some subtasks' results, re-evaluate the dependency relationships between the remaining subtasks.

Example:
\`\`\`
Task: "<some task>"

Finished subtasks:
1. "<some subtask 1>": "<some subtask result 1>
2. "<some subtask 2>": "<some subtask result 2>

Remaining subtasks:
3. "<some subtask 4>"
4. "<some subtask 5>"

Question: Considering the result of "<some subtask 1>", how are the dependency relationships affected between the remaining subtasks?
Result: "<some subtask result 1>"
Dependent subtasks:
3. "<some subtask 4>"
Remaining subtasks:
4. "<some subtask 5>"
Actions:
* add_dependency(task_number: int)
* remove_dependency(task_number: int)
Answer: ["add_dependency(4)", "remove_dependency(3)"]

(...)
\`\`\`

```

prompt:

```plain
Question: Considering the result of {current_subtask}, how are the dependency relationships affected between the remaining subtasks?
Result: {current_subtask_result}
Dependent subtasks:
3. {current_dependent_subtasks}
Remaining subtasks:
4. {remaining_subtasks}
Actions:
* add_dependency(task_number: int)
* remove_dependency(task_number: int)
Answer: {actions}
```

input: `task: str, subtasks: list[str], subtasks_finished: list[bool], subtasks_results: list[str], adjacency_lists: list[list[int]]`
output: `adjacency_lists: list[list[int]]`

## 8. If not finished, (4); else (8)

## 9. Output

```plain
Given a task, a list of subtasks and their results, write a response for the original task.

Task: {task}

Subtasks:
{i + 1}. {subtasks[i]}: {subtasks_results[i]}

Response: {task_result}
```

input: `task: str, subtasks: list[str], subtasks_results: list[str]`
output: `task_result: str`
