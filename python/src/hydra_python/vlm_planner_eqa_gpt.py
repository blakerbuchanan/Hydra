import json
from enum import Enum
from typing import List, Tuple, Literal, Any, Union, Optional, Annotated
import time
import base64

from openai import OpenAI
from hydra_python.utils import get_instruction_from_eqa_data
from pydantic import BaseModel

# client = OpenAI(
#     organization='org-9eg1dYLvm9Vnx13YZieDfE9n',
#     project='proj_rZU06lthKefMBx9rE3YGD2Um',
# )
client = OpenAI()

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def create_planner_response(frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options):

    class Goto_frontier_node_step(BaseModel):
        explanation_frontier: Annotated[str, "Explain reasoning for choosing this frontier to explore by referencing list of objects (<id> and <name>) connected to that frontier node via a link (refer to scene graph)."]
        frontier_id: frontier_node_list

    class Goto_object_node_step(BaseModel):
        explanation_room: Annotated[str, "Explain very briefly reasoning for selecting this room."]
        explanation_region: Annotated[str, "Explain very briefly reasoning for selecting this region."]
        explanation_obj: Annotated[str, "Explain very briefly reasoning for selecting this object."]
        room_id: Annotated[room_node_list, "Choose the room which contains the object you want to goto"]
        region_id:  Annotated[region_node_list, "Only select from region nodes connected to the room node (in the room)."]
        object_id: Annotated[object_node_list, "Only select from objects connected to the region node (in the region)."]
    
    class Answer(BaseModel):
        explanation_ans: Annotated[str, "Explain the reasoning for selecting the answer."]
        answer: Annotated[Answer_options, "Select the correct answer from the options."]
        explanation_conf: Annotated[str, "Explain the reasoning behind the confidence level of your answer."]
        confidence_level: Annotated[float, "Rate your level of confidence. Provide a value between 0 and 1; 0 for not confident at all and 1 for absolutely certain."]
        is_confident: Annotated[bool, "Do not use just commensense knowledge to decide confidence. Answer based on current and past observations. Choose TRUE, if you are very confident about answering the question correctly based on current and past oberservations. Choose 'FALSE', if you are uncertain of the answer and should explore more to ground your answer in the current envioronment. Clarification: This is not your confidence in choosing the next action, but your confidence in answering the question correctly."]

    class PlannerResponse(BaseModel):
        steps: List[Union[Goto_object_node_step, Goto_frontier_node_step]]
        answer: Answer
        image_description: Annotated[str, "Describe the CURRENT IMAGE. Pay special attention to features that can help answer the question or select future actions."]
        scene_graph_description: Annotated[str, "Describe the SCENE GRAPH. Pay special attention to features that can help answer the question or select future actions."]
    
    return PlannerResponse

class VLMPLannerEQAGPT:
    def __init__(self, cfg, sg_sim, question_data, output_path):
        
        self._question, self.clean_ques_ans, self.choices, self.vlm_pred_candidates = get_instruction_from_eqa_data(question_data)
        self._answer = question_data["answer"]
        self._output_path = output_path
        self._vlm_type = cfg.name
        self._use_image = cfg.use_image

        self._example_plan = '' #TODO(saumya)
        self._history = ''
        self._t = 0

        self._outputs_to_save = [f'Question: {self._question}. \n Answer: {self._answer} \n']
        self.sg_sim = sg_sim

    @property
    def t(self):
        return self._t
    
    def get_actions(self): 
        object_node_list = Enum('object_node_list', {id: name for id, name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names)}, type=str)
        if len(self.sg_sim.frontier_node_ids)> 0:
            frontier_node_list = Enum('frontier_node_list', {ac: ac for ac in self.sg_sim.frontier_node_ids}, type=str)
        else:
            frontier_node_list = Enum('frontier_node_list', {'frontier_0': 'Do not choose this option. No more frontiers left.'}, type=str)
        
        room_node_list = Enum('room_node_list', {id: name for id, name in zip(self.sg_sim.room_node_ids, self.sg_sim.room_node_names)}, type=str)
        region_node_list = Enum('region_node_list', {ac: ac for ac in self.sg_sim.region_node_ids}, type=str)
        Answer_options = Enum('Answer_options', {token: choice for token, choice in zip(self.vlm_pred_candidates, self.choices)}, type=str)
        return frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options
    
    @property
    def agent_role_prompt(self):
        scene_graph_desc = "A scene graph represents an indoor environment in a hierarchical tree structure consisting of nodes and edges/links. There are six types of nodes: building, rooms, visited areas, frontiers, objects, and agent in the environemnt. \n \
            The tree structure is as follows: At the highest level 5 is a 'building' node. \n \
            At level 4 are room nodes. There are links connecting the building node to each room node. \n \
            At the lower level 3, are region and frontier nodes. 'region' node represent region of room that is already explored. Frontier nodes represent areas that are at the boundary of visited and unexplored areas. There are links from room nodes to corresponding region and frontier nodes depicted which room they are located in. \n \
            At the lowest level 2 are object nodes and agent nodes. There is an edge from region node to each object node depicting which visited area of which room the object is located in. \
            There are also links between frontier nodes and objects nodes, depicting the objects in the vicinity of a frontier node. \n \
            Finally the agent node is where you are located in the environment. There is an edge between a region node and the agent node, depicting which visited area of which room the agent is located in."
        current_state_des = "'CURRENT STATE' will give you the exact location of the agent in the scene graph by giving you the agent node id, location, room_id and room name. Additionally, you will also be given the current view of the agent as an image. "
        
        prompt = f"You are an excellent heirarchical graph planning agent. Your goal is to navigate an unseen environment to confidently answer a multiple-choice question about the environment.\
            As you explore the environment, your sensors are building a scene graph representation (in json format) and you have access to that scene graph.  {scene_graph_desc}. {current_state_des}\
            You also have to choose the next action, one which will enable you to answer the question better. You can choose between two action types: Goto_frontier_node_step and Goto_object_node_step. \n \
            Goto_frontier_node_step: Navigates to a frontier (unexplored) node and will provide you with a new observation/image and the scene graph will be augmented/updated. \n \
            Goto_object_node_step: Navigates to a certain seen object. Choose this step in a heirarchical manner by first reasoning about which room you should be in to best answer the question, then the region where a certain collection of relevant objects are located and then the specific object."

        return prompt

    def get_current_state_prompt(self, scene_graph, agent_state):
        # TODO(saumya): Include history
        prompt = f"At t = {self.t}: \n \
            CURRENT AGENT STATE: {agent_state}. \n \
            SCENE GRAPH: {scene_graph}. \n "
        return prompt

    def get_gpt_output(self, current_state_prompt):
        
        messages=[
            {"role": "system", "content": f"AGENT ROLE: {self.agent_role_prompt}"},
            {"role": "system", "content": f"QUESTION: {self._question}"},
            {"role": "user", "content": f"CURRENT STATE: {current_state_prompt}."},
            # {"role": "user", "content": f"EXAMPLE PLAN: {self._example_plan}"} # TODO(saumya)
        ]

        if self._use_image:
            base64_image = encode_image(self._output_path / "current_img.png")
            messages.append(
                { 
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": "CURRENT IMAGE: This image represents the current view of the agent. Use this as additional information to answer the question."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                })

        frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options = self.get_actions()

        succ=False
        while not succ:
            try:
                start = time.time()
                completion = client.beta.chat.completions.parse(
                    # model="gpt-4o-mini",
                    model="gpt-4o-2024-08-06",
                    messages=messages,
                    response_format=create_planner_response(frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options),
                )
                print(f"Time taken for planning next step: {time.time()-start}s")
                plan = completion.choices[0].message
                if not (plan.refusal): # If the model refuses to respond, you will get a refusal message
                    succ=True
            except Exception as e:
                print(f"An error occurred: {e}. Sleeping for 60s")
                import ipdb; ipdb.set_trace()
                time.sleep(1)

        plan = completion.choices[0].message
        step = plan.parsed.steps[0]

        if self._use_image:
            img_desc = plan.parsed.image_description
        else:
            img_desc = ' '
        
        return step, plan.parsed.answer, img_desc, plan.parsed.scene_graph_description

    def get_next_action(self):
        # self.sg_sim.update()
        
        agent_state = self.sg_sim.get_current_semantic_state_str()
        current_state_prompt = self.get_current_state_prompt(self.sg_sim.scene_graph_str, agent_state)

        sg_desc=''
        step, answer, img_desc, sg_desc = self.get_gpt_output(current_state_prompt)

        if step.__class__.__name__ == 'Goto_object_node_step':
            target_pose = self.sg_sim.get_position_from_id(step.object_id.name)
        else:
            target_pose = self.sg_sim.get_position_from_id(step.frontier_id.name)

        # Saving outputs to file
        self._outputs_to_save.append(f'At t={self._t}: \n \
                                        Agent state: {agent_state} \n \
                                        LLM output: {step}. \n \
                                        Answer: {answer} \n \
                                        Image desc: {img_desc} \n \
                                        Scene graph desc: {sg_desc} \n \n')
        self.full_plan = ' '.join(self._outputs_to_save)
        with open(self._output_path / "llm_outputs.json", "w") as text_file:
            text_file.write(self.full_plan)

        print(f'At t={self._t}: \n {step} \n {answer}')

        self._t += 1
        return target_pose, answer.is_confident, answer.confidence_level, answer.answer.name
