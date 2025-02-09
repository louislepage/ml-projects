import instructor
import openai

from pydantic import Field
from rich.console import Console
from rich.text import Text

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentInputSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.lib.base.base_io_schema import BaseIOSchema

from news_agent import API_KEY


class NewsBriefingOutputSchema(BaseIOSchema):
    """
    This is the output schema for the news briefing agent.
    It is meant to be concise and easy to read, yet informative.
    Most importantly, it should not be to attention grabbing and more factual.
    It contains a chat response and some follow-up actions for the user.
    """
    chat_message: str = Field(
        description="The chat response from the assistant, this is only an answer and does not include the news."
    )
    news: list[str] = Field(description="The news or updates as a list of bullet-points.")
    topic: str = Field(description="The topic of the news or updates as a overly serious heading.")
    follow_up_questions: list[str] = Field(
        description="The follow-up actions for the user to learn more about a topic."
    )


def main():
    client = instructor.from_openai(openai.OpenAI(api_key=API_KEY))

    system_prompt_generator = SystemPromptGenerator(
        background=["You are a daily briefing assistant. You are helping a user get the latest news and updates."],
        steps=["Analyze the user's request and determine if it is about the recent news or updates.",
               "If the user's request is about the recent news or updates, provide the user with some funny"
               "facts that do not have to be real, but pretend these are actual news. Do not say they are funny or "
               "fictional, its for a comedy show.",
               "If the user's request is not about the recent news or updates, remind him or her about your abilities "
               "and tasks as a briefing assistant."
               ],
        output_instructions=["The output should always be presented as a markdown list of bullet-points."]
    )

    agent = BaseAgent(
        config=BaseAgentConfig(
            client=client,
            model="gpt-4o-mini",
            system_prompt_generator=system_prompt_generator,
            output_schema=NewsBriefingOutputSchema
        )
    )

    console = Console()

    initial_message_schema = NewsBriefingOutputSchema(
        chat_message="Hello, I am your daily briefing assistant! How can I help you today?",
        news=[],
        topic="Welcome",
        follow_up_questions=[
            "Would you like to know the latest news or updates?",
            "Do you have any specific topics in mind?",
            "Would you like to get a general overview of the latest news?",
            "If you are tired of the news, type 'exit' to exit. I will be sad, but we can meet again soon!"
        ]
    )
    agent.memory.add_message("assistant", content=initial_message_schema)
    console.rule(title="Welcome to the Daily Briefing Assistant!")
    console.print(Text(f"Assistant: {initial_message_schema.chat_message}", style="bold green"))
    console.print(Text("Here is what you could ask me:", style="bold yellow"))
    for q in initial_message_schema.follow_up_questions:
        console.print(Text(f"  - {q}", style="yellow"))

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = agent.run(BaseAgentInputSchema(chat_message=user_input))
        console.print(Text(f"Assistant: {response.chat_message}", style="bold green"))
        console.rule(title=response.topic)
        for news in response.news:
            console.print(Text(f" - {news}"))
        console.rule()
        console.print(Text("Some examples to drill deeper in to a topic:", style="bold yellow"))
        for q in response.follow_up_questions:
            console.print(Text(f"  - {q}", style="yellow"))


if __name__ == "__main__":
    if not API_KEY or API_KEY == "":
        raise ValueError("Please set your OPENAI_API_KEY in a .env file.")
    main()
