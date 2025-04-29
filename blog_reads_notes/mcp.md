<b> Blog link: </b> https://iaee.substack.com/p/model-context-protocol-intuitively

---

- Connecting AI systems/agents with data in a standardized protocol mannered way! (AI applications <-> data tools). Basically how applications provide context to LLMs! USB-C port of AI applications.
- RAG and agents at a high-level overview.
```
RAG: 
1) Get question from user
2) Search for information about that question
3) Combine information with question into a single prompt
4) Send prompt to LLM
5) Respond to user with LLM response
6) Go back to step 1
```
```
Agent:
1) Get question from user
2) Get list of "tools" the agent can execute
3) Create a prompt that combines the question and a description of the
tools the agent can use. Tell the LLM to choose tools to execute.
4) Send that prompt to an LLM
5) Execute the tools the LLM chose, and collect the result
6) Construct another prompt that has the result of the tools. Tell the LLM
to respond to the user based on the result of the tools.
7) Send that prompt to an LLM
8) Respond to user with LLM response
9) Go back to step 1
```
- RAG will be deterministic in terms of retrieval while agent is stochastic (an LLM might or might not use a tool for the same question asked twice!)
- <b>MCP image!!!</b>
- Claude desktop demo access screen, takes screenshots, passes that to LLM and generates action steps to be taken based on screenshoot. That is expensive and not viable always (from coder perspective)
- So, MCP on other hand creates a JSON (or Pydantic) format of the response needed for a given question so standardizing how data and models communicate
- LSP (Language Server Protocol): Connecting different coding languages (Python, R) to their tools (VS Code, Sublime) and underlying LLMs. MCP (how to connect data sources to the ecosystem of AI models) took inspiration from LSP (which is basically about how to connect programming languages to ecosystem of developer tools/IDEs)!!
- <b>LSP image</b>
- <b>LSP and MCP comparison image</b>



---
---
<b> Blog link: </b> https://huggingface.co/blog/Kseniase/mcp

---
- Orchestrations of multiple specialized models, different data sources present at different places and APIs/tools that can provide useful information!
- Limitation with tool usability - All the frameworks (including MCP) shows that models can still struggle with tool selection and usage (i.e models might not select particular tool when needed)
- Building blocks of autonomous agents
  - Profile/Knowledge - Observe and understand it's environment
  - Memory - Remember past interactions
  - Reasoning - Decide and plan it's next moves
  - Tool usage/outputs - Takes the actions
  - Self-reflection/Memory update - Reflect and learn
- MCP is not an orchestration engine or a brain! It provides a unified "toolbox" and defines how the tools are called and information is exchanged. 