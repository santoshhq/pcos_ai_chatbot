from langchain_nvidia_ai_endpoints import ChatNVIDIA ,NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langgraph.graph import START ,END ,StateGraph
from typing import List, TypedDict,Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI
import re,os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from langgraph.checkpoint.redis import RedisSaver
from langchain_core.messages.utils import trim_messages,count_tokens_approximately
from fastapi import FastAPI, HTTPException
#from supermemory import Supermemory

load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")
assert api_key is not None, "API key missing"
#memory = Supermemory(api_key="")
#gemini_model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key="")
gemini_model=ChatNVIDIA(
  model="deepseek-ai/deepseek-v3.2",
  api_key=os.getenv("NVIDIA_API_KEY")
 
)
DB_URL="redis://default:8CKkCCvcCq8DKQaFx3Pik5VkeKmsUbLv@redis-16514.crce292.ap-south-1-2.ec2.cloud.redislabs.com:16514"


def retrival_data(query:str)->List[str]:
  pc=Pinecone( api_key=os.getenv("PINECONE_API_KEY"))
  index=pc.Index(host="https://pinalcodes-l1qk32d.svc.aped-4627-b74a.pinecone.io")
  embedding =NVIDIAEmbeddings(
  model="nvidia/nv-embed-v1",
  api_key=os.getenv("NVIDIA_API_KEY")
  )
  vectorstore=PineconeVectorStore(index=index,embedding=embedding)
  retrival=vectorstore.as_retriever(search_kwargs={"k": 2})
  data=retrival.invoke(query)
  return [i.page_content for i in data]


def split_context(text):
    # split only on comma that separates list items
    parts = re.split(r"',\s*'|\",\s*\"|',\s*\"|\",\s*'", text)

    # clean quotes and spaces
    cleaned = [p.strip(" '\"\n") for p in parts if p.strip()]

    return cleaned

class BaseState(TypedDict):
  question:str
  context:List[str] 
  #good_context:List[str]
  res:str
  messages: Annotated[List[BaseMessage], add_messages] 


class verify(BaseModel):
  keep:bool

def build_graph():

    def context_retrival(state:BaseState):
      user_input=state["question"]
      res=retrival_data(user_input)
      return {"context":res}

    #def sentence_filter(state:BaseState):
      #  good_context=[]

       # prompt = ChatPromptTemplate.from_messages([
       #     ("system",
       #     "You are a Data Evaluator. The user will provide a question and a retrieved context. "
       #     "Your job is to verify whether the context is relevant. "
       #     "Return True if it should be kept, otherwise return False."
       #     ),
       #     ("human",
       #     "question: {question}\ncontext: {context}"
       #     )
       #   ])
       # chain=prompt | gemini_model.with_structured_output(verify)
        # Removed call to split_context as state["context"] is already a list of strings
        #for i in state["context"]:
        #  if chain.invoke({"question":state["question"],"context":i}).keep:
        #    good_context.append(i)
        #return {"good_context":good_context}
    

    def generate_response(state:BaseState):
      user_input=state["question"]
      model=ChatNVIDIA(
          model="mistralai/devstral-2-123b-instruct-2512",
          api_key=os.getenv("NVIDIA_API_KEY"),
          temperature=0.15)
      max_context=trim_messages(
        messages=state["messages"],
        max_tokens=500,
        strategy="last",
        token_counter=count_tokens_approximately,
      )
      prompt = ChatPromptTemplate.from_messages([

      ("system",
      """You are HealHer AI (PCOS assistant).

      Use only given context for medical info. Explain simply.
      Memory = personalization only.

      Safety: no diagnosis, no prescriptions, no certainty.
      Suggest doctor if needed.

      Behavior:
      - Symptoms → possible links only
      - Lifestyle → safe general advice
      - If not in context → say not found

      Keep answers concise. For simple/unrelated queries: ≤300 words.

      Tone: friendly, clear, supportive."""
      ),

      ("human",
      """Q: {question}

      Context:
      {context}

      """
      )
      ])
      chain=prompt | model
      res=chain.invoke({"question":user_input,"context":max_context}).content
      return {"res":res}

    graph=StateGraph(BaseState)
    graph.add_node("context_retrival",context_retrival)
    graph.add_node("generate_response",generate_response)
    #graph.add_node("sentence_filter",sentence_filter)


    graph.add_edge(START,"context_retrival")
    graph.add_edge("context_retrival","generate_response")
    #graph.add_edge("sentence_filter","generate_response")
    graph.add_edge("generate_response",END)
    return graph


class ChatRequest(BaseModel):
    question: str
    thread_id: str = "1"


class ChatResponse(BaseModel):
    res: str


app = FastAPI(title="HealHer Chatbot API", version="1.0.0")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        with RedisSaver.from_conn_string(DB_URL) as saver:
            saver.setup()
            thread_cfg = {"configurable": {"thread_id": req.thread_id}}
            inpt = {
                "question": req.question,
                "messages": [HumanMessage(content=req.question)]
            }
            result = build_graph().compile(checkpointer=saver).invoke(inpt, thread_cfg)
            return ChatResponse(res=result["res"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {exc}")
