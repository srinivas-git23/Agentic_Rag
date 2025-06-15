import gradio as gr
from smolagents import CodeAgent, LiteLLMModel
from tools import DuckDuckGoSearchTool, WeatherInfoTool, HubStatsTool
from retriever import load_guest_dataset

# Set up tools and agent
model = LiteLLMModel(
    model_id="ollama_chat/llama3.1",
    api_base="http://localhost:11434",
    api_key=None,
    num_ctx=8192,
)

search_tool = DuckDuckGoSearchTool()
weather_info_tool = WeatherInfoTool()
hub_stats_tool = HubStatsTool()
guest_info_tool = load_guest_dataset()

alfred = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool],
    model=model,
    add_base_tools=True,
    planning_interval=3
)

# Define Gradio interface function
def run_agent(query):
    return alfred.run(query)

# Launch Gradio app
gr.Interface(fn=run_agent, inputs="text", outputs="text", title="🎩 Hi, this is Agent Gala Boy! ").launch()
