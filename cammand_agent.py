from pydantic import BaseModel,Field
from langchain.tools import BaseTool
from typing import Type, TypedDict,Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage,HumanMessage,ToolMessage 
import operator
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from track_object import track_specific_object_stream


#load key
key = os.getenv("GOOGLE_API_KEY")



#Let's define our Agent
class AgentState(TypedDict):
  messages: Annotated[list[AnyMessage],operator.add]



class TrackInput(BaseModel):
    object_class: str = Field(description="should be a yolov8 class")



class CustomTrackTool(BaseTool):
  name:str="find_object"
  description:str=" This is a custom tool to find an object on video or frames sequence"
  args_schema:type[BaseModel]=TrackInput
  
  def _run(self,object_class:str):
      return track_specific_object_stream(object_class)

tool=CustomTrackTool()


class Agent:
  def __init__(self,model,tools,system=""):
    self.system=system 
    graph=StateGraph(AgentState)
    graph.add_node("llm",self.call_gemini)
    graph.add_node("action",self.take_action)
    graph.add_conditional_edges(
        "llm",
        self.exists_action,
        {True: "action",False: END}
    )
    graph.add_edge("action","llm")
    graph.set_entry_point("llm")
    self.graph=graph.compile()
    self.tools={t.name : t for t in tools}
    self.model=model.bind_tools(tools)#

  def exists_action(self,state:AgentState):
    result=state["messages"][-1]
    return len(result.tool_calls)>0 

  def call_gemini(self,state:AgentState):
    messages=state["messages"]
    if self.system:
      messages=[SystemMessage(content=self.system)] + messages 
    message=self.model.invoke(messages)
    return {"messages":[message]}

  def take_action(self,state:AgentState):
    tool_calls=state["messages"][-1].tool_calls
    results=[]
    for t in tool_calls:
      print(f"Calling: {t}")
      if not t['name'] in self.tools:
        print("\n .... bad tool name ....")
        result="bad tool name, retry"
      else:
        result=self.tools[t['name']].invoke(t['args'])
      results.append(ToolMessage(tool_call_id=t['id'],name=t['name'],content=str(result)))
      print("Back to the model!")
      return {"messages":results}
    


#Prompt

prompt="""You are smart tracking assistant.Use the track tool to loop up information.\
You are allowed to make multiple calls (either together or in sequence).\
Only look up information when you are sure of what you want.\
If you need to look up some information before asking a follow up question,you are allowed to do that!
"""
model=ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",google_api_key=key)
TrackMindBot=Agent(model,[tool],system=prompt)




messages =[HumanMessage(content="Find me chair object of  frames")]
result=TrackMindBot.graph.invoke({"messages":messages})