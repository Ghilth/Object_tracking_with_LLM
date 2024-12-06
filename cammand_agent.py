from pydantic import BaseModel,Field
from langchain.tools import BaseTool
from typing import Type, TypedDict,Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage,HumanMessage,ToolMessage 
import operator
from track_object import track_specific_object_stream





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

def find_object():
  return "Object Tracked around Manhattan at 7 a.m !!!"

def describe_video():
  return "Frames Analyzed!!!"

tools={"find_object_fn":find_object,"describe_video_fn":describe_video}
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