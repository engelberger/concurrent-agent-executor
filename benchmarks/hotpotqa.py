from dotenv import load_dotenv

load_dotenv()

import os
import json
import time
import random

import requests

from bs4 import BeautifulSoup

from react import webthink
import react.wikienv as wikienv
import react.wrappers as wrappers

from concurrent_agent_executor import BaseParallelizableTool

base_path = os.path.dirname(__file__)
model = "gpt-3.5-turbo"
throttle = 0.1


class ParallelizableSearchTool(BaseParallelizableTool):
    name = "Search"
    description = "Search for an entity on Wikipedia."
    is_parallelizable = True

    @staticmethod
    def get_page_obs(page):
        # find all paragraphs
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # find all sentence
        sentences = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        return " ".join(sentences[:5])

    def _run(self, entity: str) -> str:
        obs = ""

        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        response_text = requests.get(search_url).text
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})

        if result_divs:  # mismatch
            result_titles = [
                wikienv.clean_str(div.get_text().strip()) for div in result_divs
            ]
            obs = f"Could not find {entity}. Similar: {result_titles[:5]}."
        else:
            page = [
                p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")
            ]
            if any("may refer to:" in p for p in page):
                return self._run("[" + entity + "]")
            else:
                _page = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        _page += wikienv.clean_str(p)
                        if not p.endswith("\n"):
                            _page += "\n"
                obs = self.get_page_obs(_page)

        return obs


class ParallelizableLookupTool(BaseParallelizableTool):
    name = "Lookup"
    description = "Lookup a keyword in a Wikipedia passage."
    is_parallelizable = True

    def construct_lookup_list(self, keyword: str):
        # find all paragraphs
        if self.page is None:
            return []
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # find all sentence
        sentences = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]

        parts = sentences
        parts = [p for p in parts if keyword.lower() in p.lower()]
        return parts

    def _run(self, keyword: str) -> str:
        obs = ""

        if self.lookup_keyword != keyword:  # reset lookup
            self.lookup_keyword = keyword
            self.lookup_list = self.construct_lookup_list(keyword)
            self.lookup_cnt = 0
        if self.lookup_cnt >= len(self.lookup_list):
            obs = "No more results.\n"
        else:
            obs = (
                f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) "
                + self.lookup_list[self.lookup_cnt]
            )
            self.lookup_cnt += 1

        return obs


def init_concurrent_react():
    pass


def init_react():
    pass


env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, base_path=base_path, split="dev")


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
for i in idxs[:n]:
    r, info = webthink(env, webthink_prompt, idx=i, to_print=False)
    print(r, info)
