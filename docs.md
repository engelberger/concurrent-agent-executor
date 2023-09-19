#CAE

## Concurrent Agent Executor Base

The ConcurrentAgentExecutor located in `concurrent_agent_executor/base.py` provides a concurrent runtime environment for agents equipped with tools, which can be helpful in running multiple instances of tools in parallel.

The class essentially manages the execution lifecycle of tools called by the agent. It provides features like parallel processing, event-driven communication, execution queueing, context sharing, and state management.

### 1. Instantiation

On instantiation, the executor takes tools, an agent, and an agent's memory as the primary inputs.

Parts of the executor include:
- Event Emitters: Handles post-generation logic.
- Priority Queue: Maintains a queue of agent-tool interactions based on their priority.
- Process Pool: To manage parallel processing, a pool of processes is created with a configured number of processes to handle multiple tasks concurrently.
- Shared Global Context: A global context, shared by all running processes, is used to share the state across processes.
- Running jobs: This is a set of currently executing jobs.


### 2. Execution Control

It provides control over the executor's lifecycle through three methods: `start()`, `stop()`, and `reset()`. `start()` sets up the thread pool executor and shared memory space, `stop()` terminates all running jobs and clears the memory, and `reset()` is designed to return the executor to its initial state.

### 3. Job Management

The Executor manages jobs with several methods. For instance, `_main_thread()` fetches jobs from the queue and executes them. `_handle_call()` takes inputs and handles the execution of a job. `_start_tool()` and `_tool_callback()` manage the lifecycle of individual tool invocations.

### 4. Message Communication

Message-oriented communication is an important feature of this executor. `emit_message()` is used to send messages while `on_message()` can be used to set callbacks that listen for messages.

### 5. Agent Execution 

The executor's primary purpose is to control the execution of agents. `run_once()` is a specific method that executes the agent once and returns the agent's outputs.

### 6. Context Management

Context management is another critical task of the executor. It uses a global context so that all processes can collectively manage and update the agent's state.

The ConcurrentAgentExecutor is a powerful tool for managing the execution of complex agents with multiple tools. While it does abstract many complexities of parallel computation, understanding its precise workings can help in utilizing it most effectively. The executor allows users to do more with their agents, unlocking the potential of concurrent execution.

## Concurrent Agent Executor Queue

Located in `concurrent_agent_executor/queue.py`, the `PriorityQueueMultiGet` class is a standard priority queue with added concurrency support and multi-get function.

In the context of a Concurrent Agent Executor, this class manages the agent-tool interactions in a queue based on their priorities, ensuring operations are processed in the correct order even when dealing with concurrent events.


### 1. Instantiate the PriorityQueueMultiGet class

The `PriorityQueueMultiGet` class extends Python's built-in `PriorityQueue` class, adding multi-get functionality (the ability to get multiple items from the queue at once) and integrated threading support.

When instantiated, a threading lock is also initialized.

```python
def __init__(self):
    super().__init__()
    self.lock = threading.Lock()
```

### 2. Check if the Queue is Empty

The `empty` method is overridden to incorporate the usage of a lock providing thread safety.

```python
def empty(self):
    with self.lock:
        return super().empty()
```

### 3. Get the Size of the Queue

The `size` method also uses the lock to safely get the size of the queue when multiple threads are interacting with it.

```python
def size(self):
    with self.lock:
        return super().qsize()
```

### 4. Add Items to the Queue

The `put` method is inherited from the parent Priority Queue and is updated to include locking to ensure thread safety when adding items to the queue.

```python
def put(self, *args, **kwargs):
    with self.lock:
        super().put(*args, **kwargs)
```

### 5. Get an Item from the Queue

To get one single item, the `get` method is used. Similar to the previous methods, this also uses the lock for thread safety.

```python
def get(self, *args, **kwargs):
    with self.lock:
        return super().get(*args, **kwargs)
```

### 6. Get Multiple Items from the Queue

The `get_multiple` method retrieves multiple items (default or specified by a count) from the queue. It is important to ensure the thread's safety when fetching multiple items that the lock is used in this case as well.

```python
def get_multiple(self, count: Optional[int] = None):
    if count is None:
        count = self.size()

    with self.lock:
        items = []
        while not super().empty() and len(items) < count:
            item = super().get()
            items.append(item)
        return items
```

This `PriorityQueueMultiGet` makes sure that concurrent operations are handled correctly in priority order, and provides an operation to get multiple items at once, making it valuable for handling concurrent threaded operations in an efficient manner.

## Concurrent Agent Executor - Tools

Within the context of the Concurrent Agent Executor, located in `concurrent_agent_executor/tools.py`, the tools are classes that perform specific functionalities or tasks. These tools can execute both synchronously and asynchronously, and some of them can be parallelized.

This module extends the base tool class (`BaseParallelizableTool`) to provide several tools that deal specifically with job management in a concurrent execution context. These tools interact with other background-running jobs, allowing operations like waiting for other jobs to finish, cancelling jobs, and checking job status.

### 1. WaitTool

The `WaitTool` class is designed to wait for other tools' execution that run in the background to finish. It accepts job ids as an argument. The `WaitTool` extends the `BaseParallelizableTool` class and provides execution methods for both synchronous and asynchronous context.

#### WaitTool Schema
The `WaitToolSchema` schema describes the input that `WaitTool` expects. It expects the `job_id_or_ids`, which could be a string or a list of strings, representing the job ids that the tool should wait for.

#### Execution
The '_run' method is called to execute the tool. It returns a string indicating the job ids it is waiting for. The '_arun' method is the asynchronous variant of '_run'.

### 2. CancelTool

The `CancelTool` class is responsible for cancelling other tools that run in the background. It extends the `BaseParallelizableTool` class. The tool expects `job_id` as an argument, specifying the job id to cancel.

#### Execution
The '_run' method is called to execute the tool. However, the actual cancellation logic is not implemented and raises a `NotImplementedError`. The '_arun' method is the asynchronous variant of '_run'.

### 3. StatusTool

The `StatusTool` class checks the status of other tools running in the background. Similar to `WaitTool` and `CancelTool`, it extends the `BaseParallelizableTool` class and the tool expects `job_id` as an argument. 

#### Execution
The '_run' method is called to execute the tool. However, the actual status checking logic is not implemented and raises a `NotImplementedError`. The '_arun' method is the asynchronous variant of '_run'.

These tools provide several utility functions for managing background jobs within the context of concurrent execution. By properly configuring and using these tools with the agent executor, users can efficiently supervise and control the background jobs carried out by the tools.


## Concurrent Agent Executor initialization

Concurrent Agent Executor is a runtime system, implemented in `concurrent_agent_executor/__init__.py`, where an agent, equipped with tools, can operate concurrently. The agent might be as simple as a language model or another system that can generate and comprehend text. The tools can be anything from a text processing utility to a machine learning model.

Here's an overview of what the `initialize` function does:

It takes a language library model (llm), a list of tools, and a model name as inputs. If these are not provided, default values are used. A language library model (llm) is a tool or system that can understand and generate language. In this system, the llm used is OpenAI's language model. The tools list can contain objects from any class derived from `BaseTool` or `BaseParallelizableTool`.

Most importantly, the `initialize` function returns an instance of `ConcurrentAgentExecutor` which is a runtime encapsulation that allows the agent and the tools to operate concurrently, communicate with each other, share states, and collectively process the input and generate the output.

Here is the step-by-step process:

### 1. Create Language Model 

Below is the initialization of the language model called `llm`. If none is passed to the `initialize` function, it builds a ChatOpenAI llm instance with a default model "gpt-3.5-turbo".

```python
if llm is None:
    llm = ChatOpenAI(
        temperature=0.3,
        model=model,
    )
```

### 2. Setup Tools

Tools enhance the capabilities of the agent by performing additional operations. If no tools are provided, an empty list is initialized.

```python
if tools is None:
    tools = []
```

### 3. Conversation History and Memory Setup 

This block is used to set up the conversation history and the memory for the chat agent. The chat history is maintained as `MessagesPlaceholder` and the memory as `ConversationBufferMemory`.

```python
chat_history = MessagesPlaceholder(variable_name="chat_history")
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output",
)
```

### 4. Agent and Tools Initialization

The `ConcurrentStructuredChatAgent` is created using the llm and the tools. `input_variables` are set as "input", "agent_scratchpad" and "chat_history".

```python
agent = ConcurrentStructuredChatAgent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    memory_prompts=[chat_history],
    input_variables=["input", "agent_scratchpad", "chat_history"],
)
```

### 5. Executor Initialization

The `ConcurrentAgentExecutor` is initialized with the agent, tools, and memory. `handle_parsing_errors` is set to `True`, `early_stopping_method` is "generate" and `return_intermediate_steps` is `True`.

```python
executor = ConcurrentAgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    handle_parsing_errors=True,
    early_stopping_method="generate",
    return_intermediate_steps=True,
    **executor_kwargs,
)
```

### 6. Return Executor

Finally, the initialized executor is returned.

```python
return executor
```

To use the `initialize` function, import the `concurrent_agent_executor` module and call `initialize` with the preferred parameters. The function would return an instance of `ConcurrentAgentExecutor` with the defined properties. Now this executor instance can run the agents concurrently with the tools, allowing for efficient and parallel processing.


## ConcurrentStructuredChatAgent class

concurrent_agent_executor/structured_chat/base.py
ConcurrentStructuredChatAgent is a base class for structured chat agents. An instance of this class represents a chat agent that carries out structured conversation. By structured conversation, we mean that the agent processes input, decides what tool to use while considering multiple parallel events, processing the output, and deciding next steps. The agent typically interacts with users and tools under different interaction types.
The `ConcurrentStructuredChatAgent` is a base class for structured chat agents which process user's input during a chat conversation, decide and choose the appropriate tool(s) to handle those inputs, process the actions of those tools, and decide on the next course of action. It allows chat agents to carry out structured conversations by logically making decision on what next action to execute from the set of given input variables, using the llm_chain and tools configured for them. The class provides default behaviour as well as customizability for more complex interactions.

### What the class does

During a chat, the agent, upon receiving user's input, first processes the input and uses the `llm_chain` or `system_llm_chain` (based on the type of interaction with user or tools respectively) to decide action. It then uses the tool that it has decided and processes the tool's output by parsing it using the `output_parser`. Then it follows the same loop until the conversation ends.

Generally, the `system_llm_chain` is used to model when the agent interacts with the tool, and the `llm_chain` is used for the user-agent interaction.

This makes the agent flexible, allowing it to interactively process the input, execute the action using the tool, then parse the output interactively and decide on the next step.

### Core methods and attributes

0. `system_llm_chain`: This is the chain of tools the agent executes, lineage logged models(LLMChain) to be specific, to make a decision or to process the user's input.

1. `__init__`: The constructor which initializes the LLM Chains and other necessary parameters.

2. `create_prompt`: Formats the chat prompt template for structured chat.

3. `from_llm_and_tools`: This class method is used for creating an agent from a given LLM model and tools. This setup involves specifying the conversations' start and end, and defining instructions for the user.

4. `plan`: Based on the input and the intermediate steps in the conversation, this function decides the next step and takes the corresponding action.

5. `output_parser`: This attribute indicates the default function used for parsing output interactively.

### Overridable attributes for customization

1. `system_prefix`: Defines a prefix to append to the system message.

2. `observation_prefix`: Defines a prefix to append to the observation message.

3. `llm_prefix`: Defines a prefix to append to the llm thoughts feed.

## Structured Chat Output Parser

The `StructuredChatOutputParser` class in `output_parser.py` is a mechanism developed to interpret the structured chat generated by agents in the system.

The idea is simple: Agents, upon receiving input and giving output, follow a structured chat format. The `StructuredChatOutputParser` takes the text output from these agents, interprets the specified format, and returns an 'action' that represents the output in a standardized way that the system can understand and work with.

The class primarily handles two general outputs:
1. `AgentActionWithId`: This is an action taken by an agent. It includes the action name, any additional input to the action, the original message text, and a newly generated ID for the action (generated by the `generate_id` function from`human_id` module).
2. `AgentFinish`: This signifies that the agent has finished its output.

Below is a step-by-step breakdown of how the class functions:

### 1. Format Instructions Extraction

The method `get_format_instructions` returns the formatting instructions. The corresponding instructions are defined in the `langchain.agents.structured_chat.prompt` module under `FORMAT_INSTRUCTIONS`.

### 2. Output Parsing

The `parse` method takes in a string text as input and returns either an `AgentActionWithId` or an `AgentFinish` object depending on the content.

It performs the following steps:

- Using regular expressions, it searches for text enclosed in triple backticks (```) in the input. This block is supposed to contain JSON data representing the action.
- If found, the data is parsed to create an action. If multiple actions are found, it chooses the first one.
- If the action's name is "Final Answer", an `AgentFinish` object is created.
- For any other action, an `AgentActionWithId` object is created.
- If no block with triple backticks is found, create an `AgentFinish` object with the complete text.
- Raise an `OutputParserException` if any error occurs during the parsing process.

### StructuredChatOutputParserWithRetries

The `StructuredChatOutputParserWithRetries` class is an extension of `StructuredChatOutputParser`. The difference is that `StructuredChatOutputParserWithRetries` uses an `OutputFixingParser` to try fixing errors in the output before parsing it. 

This variant provides a resiliency mechanism to parse the output successfully even if there are some minor issues or mistakes made by the agent in structuring the chat output.

It follows the same steps as `StructuredChatOutputParser` with one additional step before the actual parsing: if an `OutputFixingParser` is defined, the class uses it to preprocess and potentially fix the text input, ensuring a better chance of successful parsing.

In order to create an instance of `StructuredChatOutputParserWithRetries`, use the `from_llm` class method which accepts an optional `llm` object and a `base_parser` object as arguments.

The returned object depends on the provided arguments:
- If 'llm' is provided, an `OutputFixingParser` is created using 'llm' and the 'base_parser'. If no 'base_parser' is given, `StructuredChatOutputParser` is used as default.
- If only 'base_parser' is provided, the returned object uses it as the base parser.
- If no arguments are given, the default `StructuredChatOutputParser` is used.