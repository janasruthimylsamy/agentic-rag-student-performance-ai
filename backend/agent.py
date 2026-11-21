
import json


MODEL_NAME = "openai/gpt-oss-120b"


def run_agent(
    user_query,
    groq_client,
    groq_tools,
    execute_agent_tool,
    system_prompt,
    max_iterations=5
):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    for _ in range(max_iterations):

        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=groq_tools,
            tool_choice="auto",
            temperature=0
        )

        assistant_message = response.choices[0].message

        if not assistant_message.tool_calls:
            return assistant_message.content

        messages.append(
            assistant_message.model_dump(exclude_none=True)
        )

        for tool_call in assistant_message.tool_calls:

            tool_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            tool_result = execute_agent_tool(
                tool_name,
                arguments
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    tool_result,
                    default=str
                )
            })

    return "Agent reached the maximum number of tool iterations."


def run_rag_agent(
    user_query,
    groq_client,
    groq_tools,
    execute_agent_tool,
    system_prompt,
    retrieve_policy,
    max_iterations=5
):
    retrieved_docs = retrieve_policy(user_query)

    policy_context = "\n\n".join(
        doc["text"] for doc in retrieved_docs
    )

    enhanced_prompt = (
        system_prompt
        + "\n\nRETRIEVED POLICY DOCUMENTS:\n"
        + policy_context
        + "\n\nUse retrieved policy documents "
        "for policy-related questions. "
        "Do not invent policy rules."
    )

    return run_agent(
        user_query=user_query,
        groq_client=groq_client,
        groq_tools=groq_tools,
        execute_agent_tool=execute_agent_tool,
        system_prompt=enhanced_prompt,
        max_iterations=max_iterations
    )
