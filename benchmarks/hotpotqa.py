from dotenv import load_dotenv

load_dotenv()

import os
import json
import random
import time

import react.wikienv as wikienv
import react.wrappers as wrappers
from react import webthink

base_path = os.path.dirname(__file__)

env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, base_path=base_path, split="dev")
env = wrappers.LoggingWrapper(env)


folder = "prompts"
prompt_file = "prompts_naive.json"
with open(
    os.path.join(base_path, folder, prompt_file),
    "r",
) as f:
    prompt_dict = json.load(f)

webthink_examples = prompt_dict["webthink_simple6"]
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples

idxs = list(range(7405))
random.Random(233).shuffle(idxs)

n = 1

rs = []
infos = []
# old_time = time.time()
for i in idxs[:n]:
    r, info = webthink(env, webthink_prompt, idx=i, to_print=False)
    print(r)
    print(info)

    # rs.append(info["em"])
    # infos.append(info)
    # print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
    # print("-----------")
    # print()
