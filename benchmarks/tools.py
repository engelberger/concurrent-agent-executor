from dotenv import load_dotenv

load_dotenv()

import traceback

from contextlib import contextmanager
from typing import Optional

import requests

from bs4 import BeautifulSoup

from concurrent_agent_executor import BaseParallelizableTool, initialize
from concurrent_agent_executor.utils import tail


class ParallelizableSearchTool(BaseParallelizableTool):
    name = "search"
    description = "Search for a Wikipedia page."
    is_parallelizable = True

    entity: Optional[str] = None

    @staticmethod
    def get_search_url(entity: str) -> str:
        # ? NOTE: This is a hacky
        entity_url_encoded = entity.replace(" ", "+")
        return f"https://en.wikipedia.org/w/index.php?search={entity_url_encoded}"

    @staticmethod
    def get_page_observation(page: str) -> str:
        # find all paragraphs
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # find all sentence
        sentences = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        return " ".join(sentences[:5])

    @staticmethod
    def clean_str(p: str) -> str:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

    @property
    def result_titles(self) -> list:
        return self.global_context[self.entity].get("result_titles", [])

    def set_result_titles(self, result_titles: list):
        return self.global_context[self.entity].update({"result_titles": result_titles})

    @property
    def page(self) -> str:
        return self.global_context[self.entity].get("page", "")

    def set_page(self, page: str):
        return self.global_context[self.entity].update({"page": page})

    @property
    def lookup_keyword(self) -> str:
        return self.global_context[self.entity].get("lookup_keyword", "")

    def set_lookup_keyword(self, lookup_keyword: str):
        return self.global_context[self.entity].update(
            {"lookup_keyword": lookup_keyword}
        )

    @property
    def lookup_list(self) -> list:
        return self.global_context[self.entity].get("lookup_list", [])

    def set_lookup_list(self, lookup_list: list):
        return self.global_context[self.entity].update({"lookup_list": lookup_list})

    @property
    def lookup_cnt(self) -> int:
        return self.global_context[self.entity].get("lookup_cnt", 0)

    def set_lookup_cnt(self, lookup_cnt: int):
        return self.global_context[self.entity].update({"lookup_cnt": lookup_cnt})

    @contextmanager
    def _context(self, entity: str):
        try:
            self.entity = entity
            self.global_context[self.entity] = {}
            self.global_context["_current_entity"] = self.entity
            yield
        finally:
            self.entity = None

    def _run_inner(self, entity: str) -> str:
        observation = ""

        with self._context(entity):
            response_text = requests.get(self.get_search_url(entity)).text
            soup = BeautifulSoup(response_text, features="html.parser")
            result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})

            if result_divs:  # mismatch
                result_titles = [
                    self.clean_str(div.get_text().strip()) for div in result_divs
                ]

                self.set_result_titles(result_titles)

                observation = f"Could not find {entity}. Similar: {result_titles[:5]}. Try again with a similar entity."
            else:
                page = [
                    p.get_text().strip()
                    for p in soup.find_all("p") + soup.find_all("ul")
                ]
                if any("may refer to:" in p for p in page):
                    # NOTE: this might be a quirk of the wikipedia search api
                    return self._run("[" + entity + "]")
                else:
                    _page = ""
                    for p in page:
                        if len(p.split(" ")) > 2:
                            _page += self.clean_str(p)

                            if not p.endswith("\n"):
                                _page += "\n"

                    self.set_page(_page)
                    self.set_lookup_keyword("")
                    self.set_lookup_list([])
                    self.set_lookup_cnt(0)

                    observation = self.get_page_observation(page)

        return observation

    def _run(self, entity: str) -> str:
        try:
            return self._run_inner(entity)
        except Exception as e:
            print(e, traceback.format_exc(), sep="\n")
            return f"error: {e}"


class LookupTool(BaseParallelizableTool):
    name = "lookup"
    description = "Lookup a keyword in the current Wikipedia page."
    is_parallelizable = True

    @property
    def result_titles(self) -> list:
        return self.global_context[self.entity].get("result_titles", [])

    def set_result_titles(self, result_titles: list):
        return self.global_context[self.entity].update({"result_titles": result_titles})

    @property
    def page(self) -> str:
        return self.global_context[self.entity].get("page", "")

    def set_page(self, page: str):
        return self.global_context[self.entity].update({"page": page})

    @property
    def lookup_keyword(self) -> str:
        return self.global_context[self.entity].get("lookup_keyword", "")

    def set_lookup_keyword(self, lookup_keyword: str):
        return self.global_context[self.entity].update(
            {"lookup_keyword": lookup_keyword}
        )

    @property
    def lookup_list(self) -> list:
        return self.global_context[self.entity].get("lookup_list", [])

    def set_lookup_list(self, lookup_list: list):
        return self.global_context[self.entity].update({"lookup_list": lookup_list})

    @property
    def lookup_cnt(self) -> int:
        return self.global_context[self.entity].get("lookup_cnt", 0)

    def set_lookup_cnt(self, lookup_cnt: int):
        return self.global_context[self.entity].update({"lookup_cnt": lookup_cnt})

    @contextmanager
    def _context(self):
        try:
            self.entity = self.global_context["_current_entity"]
            yield
        finally:
            self.entity = None

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

    def _run_inner(self, keyword: str) -> str:
        observation = ""

        with self._context(keyword):
            if self.lookup_keyword != keyword:  # reset lookup
                self.set_lookup_keyword(keyword)
                self.set_lookup_list(self.construct_lookup_list(keyword))
                self.set_lookup_cnt(0)

            if self.lookup_cnt >= len(self.lookup_list):
                observation = "No more results.\n"
            else:
                observation = (
                    f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) "
                    + self.lookup_list[self.lookup_cnt]
                )
                self.set_lookup_cnt(self.lookup_cnt + 1)

        return observation

    def _run(self, keyword: str) -> str:
        try:
            return self._run_inner(keyword)
        except Exception as e:
            print(e, traceback.format_exc(), sep="\n")
            return f"error: {e}"


def main():
    executor = initialize(
        tools=[
            ParallelizableSearchTool(),
            LookupTool(),
        ],
    )

    prompt = "Solve a question answering task using the tools you have available: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"

    try:
        executor.start()
        run = executor.run_once({"input": prompt})
        outputs = tail(run)
        print(outputs)
        print(f"Used {run.intermediate_steps}")
        print(f"Took {run.running_time} seconds")
        print(f"LLM took {run.llm_generation_time} seconds")
    finally:
        executor.stop()


if __name__ == "__main__":
    main()
