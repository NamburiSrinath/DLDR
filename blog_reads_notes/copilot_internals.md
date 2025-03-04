<b>Blog post:</b> https://thakkarparth007.github.io/copilot-explorer/posts/copilot-internals.html

- Copilot can predict what variables, functions I plan to use only if it sends something relevant to the Codex model as part of it's prompt.
- It gets access to 
  * entry point where the cursor is present
  * language and relative path of the folder
  * last 20 files which has been written in same language
  * some configurations on FIM (filling-in-middle) code.

Inline/Ghost text editor
- Client should not request the model if the client is typing too fast, not too many requests (for cost efficiency) and caches the model responses. 
- After generating the prompt, it computes "contextual score" to understand whether to invoke the model or not. Because invoking a model around ] or ) makes less sense compared with [ or (

Copilot Panel
- Here the number of requests are more and there's no contextual filtering score as the user explicitly wants code completion.

Telemetry
- Github measures whether their changes have been accepted or rejected not by mere accepted button but by looking at edits that are done after 15 sec to 15min and checking if the accepted code is still present in the codebase. Minor changes are not taken into consideration for this metrics. 
- The telemetry also sends code snippets of whether it has been accepted or rejected as part of future improvements and one can opt out if needed!
- Most likely they are using a smaller model (12B) thus being cost effective.
