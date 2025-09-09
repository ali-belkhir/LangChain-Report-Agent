from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from schemas import ReportOutput, ReportSummary, PriceEvolution
from tools import tools, generate_word_report



# Initialize the FastAPI app for the AI service.
app = FastAPI(
    title="LangChain AI Service",
    description="Processes task descriptions and generates structured project data.",
)

# Define a Pydantic model for the incoming request.
class TaskPayload(BaseModel):
    task_description: str

# Initialize the Ollama model for the agent.
llm = ChatOllama(model="mistral", temperature=0)

# Define the output parser based on our Pydantic model.
parser = PydanticOutputParser(pydantic_object=ReportOutput)

# Create the prompt template for the LLM agent.
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a report generator AI. "
        "Your job is to extract content from the given URL using the 'get_website_content' tool, "
        "analyze it, and write a clear structured report. "
        "You must return the output in **strict JSON format** that matches the schema:\n"
        "- url: string\n"
        "- report_summary: object with title, date, description, prices (energy, price, evolution)\n"
        "- word_file_path: string\n\n"
        "Do NOT include any explanations, code snippets, or Python functions in your output. "
        "Only return valid JSON."
        "\n\n{format_instructions}"
    )),
    ("placeholder", "{chat_history}"),
    ("user", "{task_description}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create the agent executor.
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Define the endpoint for processing tasks.
@app.post("/process-task")
async def process_task_with_ai(payload: TaskPayload):
    if not payload.task_description:
        raise HTTPException(status_code=400, detail="Task description cannot be empty.")
    
    try:
        # Invoke the LangChain agent executor
        response = agent_executor.invoke({
            "task_description": payload.task_description,
            "format_instructions": parser.get_format_instructions()
        })
        
        # Parse AI response
        final_output_text = response["output"]
        result = parser.parse(final_output_text)

        # Generate Word file from the report 
        word_path = generate_word_report(result.report_summary)
        result.word_file_path = word_path

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during AI processing: {str(e)}")