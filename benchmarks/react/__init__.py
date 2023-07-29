import os
from typing import Optional
import gym
import openai
import requests

openai.api_key = os.environ["OPENAI_API_KEY"]


def llm(prompt: str, stop: list[str] = ["\n"]) -> str:
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop,
    )
    return response["choices"][0]["text"]


def step(env: gym.Env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1


def webthink(
    env: gym.Env, prompt: str, idx: Optional[int] = None, to_print: bool = True
):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])

        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print("ohh...", thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split("\n")[0]
            action = llm(
                prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]
            ).strip()

        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace("\\n", "")
        step_str = (
            f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        )
        prompt += step_str
        if to_print:
            print(step_str)
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    if to_print:
        print(info, "\n")
    info.update({"n_calls": n_calls, "n_badcalls": n_badcalls, "traj": prompt})
    return r, info
