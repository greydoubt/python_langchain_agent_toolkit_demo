import os
os.environ["LANGCHAIN_TRACING"] = "true"

from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType

llm = OpenAI(temperature=0)

from langchain.tools import StructuredTool

def multiplier(a: float, b: float) -> float:
    """Multiply the provided floats."""
    return a * b

tool = StructuredTool.from_function(multiplier)

# Structured tools are compatible with the STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION agent type. 
agent_executor = initialize_agent([tool], llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent_executor.run("What is 53141 times 17?")


# EXPECTED_OUTPUT: '53141 times 17 is 903397'


'''
> Entering new AgentExecutor chain...

Thought: I need to multiply 3 and 4
Action:
```
{
  "action": "multiplier",
  "action_input": {"a": 3, "b": 4}
}
```

Observation: 12
Thought: I know what to respond
Action:
```
{
  "action": "Final Answer",
  "action_input": "53141 times 17 is 903397"
}
```

> Finished chain.

'''
