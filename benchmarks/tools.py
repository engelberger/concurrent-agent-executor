from concurrent_agent_executor import BaseParallelizableTool


class ParallelizableSearchTool(BaseParallelizableTool):
    name = "search"
    description = "Search for a Wikipedia page."
    is_parallelizable = True


class LookupTool(BaseParallelizableTool):
    name = "lookup"
    description = "Lookup a keyword in the current Wikipedia page."
    is_parallelizable = False
