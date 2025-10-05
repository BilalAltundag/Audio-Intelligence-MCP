import asyncio
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
import os


# MCP server config: Stdio transport ile app.py'yi subprocess olarak başlat
config = {
    "audio_mcp": {
        "transport": "stdio",
        "command": "python",  # Python interpreter
        "args": ["main.py"]    # Server script'i
    }
}

# Client'ı config ile oluştur (stdio otomatik yönetir)
client = MultiServerMCPClient(config)

# Tool'ları async yükle
async def load_tools():
    return await client.get_tools()

tools = asyncio.run(load_tools())
print(tools)
# Gemini modeli (Flash), system message ile
model = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
    temperature=0.7,
    google_api_key="api-key"
)
print(model)
# System message'i prompt ile entegre et (Gemini system_message desteklemez, prompt kullan)
system_message = "You are a helpful audio processing assistant. Use the provided MCP tools to handle audio tasks like transcription, analysis, etc."

# ReAct agent'ı oluştur (tool calling için)
agent = create_react_agent(model, tools)

# Test için basit bir chain: System message + user input
async def run_agent(user_input: str):
    messages = [
        SystemMessage(content=system_message),
        {"role": "user", "content": user_input}
    ]
    response = await agent.ainvoke({"messages": messages})
    return response["messages"][-1].content  # Son yanıt

# Örnek kullanım
if __name__ == "__main__":
    user_input = "Label these audio files: ['audio/speech-94649.wav', 'audio/speech-dramatic-female-38105.wav']"
    print("Agent Response:", asyncio.run(run_agent(user_input)))