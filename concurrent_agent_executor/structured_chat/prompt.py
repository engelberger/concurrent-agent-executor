"""Prompt strings for the structured chat agent."""

# flake8: noqa

# pylint: disable=line-too-long
PREFIX = """Respond to the human as helpfully and accurately as possible. You have access to the following tools:"""

# pylint: disable=line-too-long
FORMAT_INSTRUCTIONS = """Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}}}
```"""

# pylint: disable=line-too-long
SUFFIX = """Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Sometimes you will schedule tools to RUN IN THE BACKGROUND; in those cases, the system will tell you when they are done running. You can have multiple tools running in the background. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
Thought:"""

RESULT = """Result: {result}"""

START_BACKGROUND_JOB = "Running {tool_name} in the background with job_id {job_id}"
