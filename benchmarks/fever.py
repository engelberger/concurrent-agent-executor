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
env = wrappers.FeverWrapper(env, base_path=base_path, split="dev")
env = wrappers.LoggingWrapper(env)


folder = "prompts"
prompt_file = "fever.json"
with open(os.path.join(base_path, folder, prompt_file), "r") as f:
    prompt_dict = json.load(f)

webthink_prompt = prompt_dict["webthink_simple3"]

idxs = list(range(7405))
random.Random(233).shuffle(idxs)

n = 3
rs = []
infos = []
old_time = time.time()
for i in idxs[:n]:
    r, info = webthink(i, to_print=True)
    rs.append(info["em"])
    infos.append(info)
    print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
    print("-----------")
    print()
