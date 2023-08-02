import json

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from graph_of_thought.prompt import determine_subtasks_prompt


def _make_nodes_hash(arr: list[str]) -> dict[str, int]:
    node_hashes = {}
    for i in range(len(arr)):
        node_hashes[arr[i]] = i
    return node_hashes


def _check_nodes(nodes: list[str], edges: tuple[str, str]) -> None:
    node_hashes = _make_nodes_hash(nodes)
    for edge in edges:
        if edge[0] not in node_hashes or edge[1] not in node_hashes:
            missing = set([edge[0]]) if edge[0] not in node_hashes else set([edge[1]])
            raise ValueError(
                f"Unknown node. There is an unknown node in the supplied edges ({', '.join(missing) or 'unknown'})."
            )


def parallel_toposort(nodes: list[str], edges: tuple[str, str]) -> list[list[str]]:
    _check_nodes(nodes, edges)

    visited: set[str] = set()
    _sorted: list[list[str]] = []

    while len(visited) < len(nodes):
        _next = [
            node
            for node in nodes
            if node not in visited
            and all(
                dependency in visited
                for dependency in [edge[0] for edge in edges if edge[1] == node]
            )
        ]

        if len(_next) == 0:
            raise ValueError("Circular dependency detected")

        for node in _next:
            visited.add(node)
        _sorted.append(_next)

    return _sorted


def determine_subtasks(llm: ChatOpenAI, task: str) -> list[str]:
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=determine_subtasks_prompt,
            input_variables=["task"],
        ),
        # ! TODO: Add output parser
        # output_key="subtasks",
        # output_parser=...,
    )

    subtasks = json.loads(chain.predict(task=task))
    assert isinstance(subtasks, list)

    return subtasks


def determine_subtasks_interdependencies(
    llm: ChatOpenAI, task: str, subtasks: list[str]
) -> list[list[str]]:
    # messages = [
    #     SystemMessagePromptTemplate.from_template(template),
    #     *_memory_prompts,
    #     HumanMessagePromptTemplate.from_template(human_message_template),
    # ]
    # ChatPromptTemplate(input_variables=[], messages=messages)

    memory = ConversationBufferMemory(
        ai_prefix="",
        human_prefix="",
    )
    chain = ConversationChain(
        llm=llm,
        # prompt=
    )

    raise NotImplementedError


def run(task: str):
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)  # type: ignore

    subtasks = determine_subtasks(llm, task)
    print(subtasks)
    adjacency_list = determine_subtasks_interdependencies(llm, task, subtasks)
    print(adjacency_list)
