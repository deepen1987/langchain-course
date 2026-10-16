# Algorithm: `run_agent(question)`: tool-calling agent loop

Source: `src/langchain_course/__pycache__/1_agent_loop_langchain_tool_calling.py`

**Input:** `question` (string)
**Constants:** `MODEL`, `URL` (from `.env`), `MAX_ITERATIONS = 5`
**Output:** the final answer text, or `None` if it runs out of iterations

## Setup

1. **Register tools:** `tools = [get_product_price, apply_discount]`
2. **Build a lookup table:** `tool_dict = { tool.name → tool }`
   *(turns the name the model gives back into a function you can call)*
3. **Create the LLM:** `llm = init_chat_model(MODEL, provider="ollama", base_url=URL)`
4. **Bind tools:** `llm_with_tools = llm.bind_tools(tools)`
   *(every request now carries the tool schemas, so the model can reply with tool calls)*
5. **Seed the history** `messages`:
   - `SystemMessage`: role, tool list, strict rules (never guess, get the price before discounting, never do the maths yourself)
   - `HumanMessage(question)`

## Loop: `for i = 1 … MAX_ITERATIONS`

6. **Think:** `ai_message = llm_with_tools.invoke(messages)`
7. **Read tool calls:** `tool_calls = ai_message.tool_calls`
8. **Stop if done:** if `tool_calls` is empty, **return `ai_message.content`** *(final answer)*
9. **Take the first tool call only:**
   - `name = tool_call["name"]`
   - `args = tool_call["args"]` (defaults to `{}`)
   - `id = tool_call["id"]`
10. **Validate:** `tool = tool_dict[name]`. If it's missing, **raise `ValueError`**
11. **Act:** `observation = tool.invoke(args)`
12. **Update the history** (the model has no memory, so this is how it knows what happened):
    - `messages.append(ai_message)` *(the model's request)*
    - `messages.append(ToolMessage(str(observation), tool_call_id=id))` *(the result, linked to the request by its id)*
13. Go back to step 6

## Exit

14. If the loop finishes, print "Max iterations reached" and **return `None`**

## Flow diagram

```
question
   │
   ▼
[System + Human messages]
   │
   ▼
┌─► LLM.invoke(messages) ──► tool_calls?
│                              │
│                     no ──────┴────── yes
│                     │                 │
│              return answer     run tool(args)
│                                       │
│                     append ai_message + ToolMessage(result, id)
│                                       │
└──────────── (i < MAX_ITERATIONS) ◄────┘
                     │ else
                     ▼
               return None
```

## Example run

```
Q: What is the price of the laptop with a gold discount?

Iteration 1/5 → get_product_price({'product': 'laptop'})                   → 999.99
Iteration 2/5 → apply_discount({'price': 999.99, 'discount_tier': 'gold'}) → 849.9915
Iteration 3/5 → AI Response: The price of the laptop with a gold discount is 849.99.
```

## Key ideas

- **Agent = loop:** think → act → observe → repeat, until there are no more tool calls.
- **The model never runs code.** It only *asks* for a call, and your code runs it.
- **`messages` is the agent's only memory.** Skip step 12 and it asks for the same tool forever.
- **`tool_call_id`** links each `ToolMessage` to the call that asked for it.
- **`MAX_ITERATIONS`** is a safety stop, not part of the logic.

## Quirks in the current code

- Product lookup is case-sensitive: `"Laptop"` returns `0.0`.
- An unknown tool raises an exception instead of returning an error message to the model.
- Only the first tool call per iteration is run. Real agents often run every call in `ai_message.tool_calls`.
