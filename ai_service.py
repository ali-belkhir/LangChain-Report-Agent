from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from schemas import ReportOutput, ReportSummary, PriceEvolution
from tools import generate_word_report, get_website_content, retrieve_relevant_context
from langchain_core.tools import Tool

# Wrap your Python functions into LangChain Tool objects
website_tool = Tool(
    name="get_website_content",
    func=get_website_content,
    description="Extracts raw text content from a given website URL"
)

rag_tool = Tool(
    name="retrieve_relevant_context",
    func=retrieve_relevant_context,
    description="Retrieve only relevant chunks of text from the RAG index"
)

report_tool = Tool(
    name="generate_word_report",
    func=generate_word_report,
    description="Generates a Word report file from a structured summary"
)


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
        "You are a report generator AI agent.\n"
        "1. Use the 'get_website_content' tool to extract the raw content from the URL.\n"
        "2. Use the RAG pipeline to retrieve only the relevant informations based on the task.\n"
        "3. Analyze this context and generate a clear structured report.\n\n"
        "The output must be in strict JSON format matching the schema:\n"
        "- url: string\n"
        "- report_summary: {{ title, date, description }}\n"
        "- word_file_path: string\n\n"
        "Finally, call 'generate_word_report' tool to create the Word file.\n\n"
        "Only return valid JSON.\n\n{format_instructions}"
    )),
    ("placeholder", "{chat_history}"),
    ("user", "{task_description}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create the agent executor.
agent = create_tool_calling_agent(
    llm=llm,
    tools=[website_tool, rag_tool, report_tool],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[website_tool, rag_tool, report_tool],
    verbose=True
)

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