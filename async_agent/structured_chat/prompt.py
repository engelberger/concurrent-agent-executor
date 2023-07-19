# flake8: noqa

PREFIX = """Respond to the human as helpfully and accurately as possible. You have access to the following tools:"""

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

SUFFIX = """Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Reminder to use the _Wait tool to WAIT FOR OTHER TOOLS THAT RUN IN THE BACKGROUND. Sometimes you will schedule TOOLS/JOBS TO RUN IN THE BACKGROUND, AND THE SYSTEM WILL TELL YOU WHEN THEY ARE DONE RUNNING. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
Thought:"""

RESULT = """Result: {result}"""
