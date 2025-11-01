# from openai import OpenAI
# client = OpenAI()

# response = client.responses.create(
#     model="gpt-5",
#     input="Write a one-sentence bedtime story about a unicorn."
# )

# print(response.output_text)

# from openai import OpenAI
# client = OpenAI()

# stream = client.responses.create(
#     model="gpt-5",
#     input=[
#         {
#             "role": "user",
#             "content": "Say 'double bubble bath' ten times fast.",
#         },
#     ],
#     stream=False,
# )

# for event in stream:
#     print(event)
# ------------------------------------------------------------------
# from openai import OpenAI
# client = OpenAI()

# stream = client.responses.create(
#     model="gpt-5",
#     input=[
#         {
#             "role": "user",
#             "content": "Say 'double bubble bath' ten times fast.",
#         },
#     ],
#     stream=True,
# )

# for event in stream:
    # print(event)


from agents import Agent, Runner
import asyncio

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
)


async def main():
    while True:
        result = await Runner.run(triage_agent, input=input("You: "))
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())