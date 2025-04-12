import json
from enum import Enum
from typing import List, Tuple, Literal, Any, Union, Optional, Annotated
import time
import base64

from openai import OpenAI
from hydra_python.utils import get_latest_image
from pydantic import BaseModel
import numpy as np
import click 

client = OpenAI()

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def create_planner_response(frontier_node_list, room_node_list, region_node_list, object_node_list, use_image=True):

    class Goto_frontier_node_step(BaseModel):
        explanation_frontier: str
        frontier_id: frontier_node_list

    class Goto_object_node_step(BaseModel):
        explanation_room: str
        # explanation_region: str
        explanation_obj: str
        room_id: room_node_list
        # region_id: region_node_list
        object_id: object_node_list
    
    class Answer(BaseModel):
        explanation_ans: str
        answer: str
        explanation_conf: str
        confidence_level: float
        is_confident: bool

    class PlannerResponse(BaseModel):
        steps: List[Union[Goto_object_node_step, Goto_frontier_node_step]]
        answer: Answer
        image_description: str
        scene_graph_description: str
    
    class PlannerResponseNoFrontiers(BaseModel):
        steps: List[Goto_object_node_step]
        answer: Answer
        image_description: str
        scene_graph_description: str
    
    class PlannerResponseNoImage(BaseModel):
        steps: List[Union[Goto_object_node_step, Goto_frontier_node_step]]
        answer: Answer
        scene_graph_description: str
    
    class PlannerResponseNoFrontiersNoImage(BaseModel):
        steps: List[Goto_object_node_step]
        answer: Answer
        scene_graph_description: str
    
    if use_image:
        if frontier_node_list is None:
            return PlannerResponseNoFrontiers
        else:
            return PlannerResponse
    else:
        if frontier_node_list is None:
            return PlannerResponseNoFrontiersNoImage
        else:
            return PlannerResponseNoImage
    

class VLMPLannerOpenEQAGPT:
    def __init__(self, cfg, sg_sim, question, answer, output_path, evaluation_prompt):
        
        self._question = question
        self._answer = answer
        self._output_path = output_path
        self._vlm_type = cfg.name
        self._use_image = cfg.use_image

        self._example_plan = '' #TODO(saumya)
        self._history = ''
        self.full_plan = ''
        self._t = 0
        self._add_history = cfg.add_history

        self._outputs_to_save = [f'Question: {self._question}. \n Answer: {self._answer} \n']
        self.sg_sim = sg_sim
        self.evaluation_prompt = evaluation_prompt

    @property
    def t(self):
        return self._t
    
    def get_actions(self): 
        object_node_list = Enum('object_node_list', {id: name for id, name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names)}, type=str)
        if len(self.sg_sim.frontier_node_ids)> 0:
            frontier_node_list = Enum('frontier_node_list', {ac: ac for ac in self.sg_sim.frontier_node_ids}, type=str)
        else:
            # frontier_node_list = Enum('frontier_node_list', {'frontier_0': 'Do not choose this option. No more frontiers left.'}, type=str)
            frontier_node_list = None
        
        room_node_list = Enum('room_node_list', {id: name for id, name in zip(self.sg_sim.room_node_ids, self.sg_sim.room_node_names)}, type=str)
        region_node_list = Enum('region_node_list', {ac: ac for ac in self.sg_sim.region_node_ids}, type=str)
        return frontier_node_list, room_node_list, region_node_list, object_node_list
    
    @property
    def agent_role_prompt(self):
        scene_graph_desc = "A scene graph represents an indoor environment in a hierarchical tree structure consisting of nodes and edges/links. There are six types of nodes: building, rooms, visited areas, frontiers, objects, and agent in the environemnt. \n \
            The tree structure is as follows: At the highest level 5 is a 'building' node. \n \
            At level 4 are room nodes. There are links connecting the building node to each room node. \n \
            At the lower level 3, are region and frontier nodes. 'region' node represent region of room that is already explored. Frontier nodes represent areas that are at the boundary of visited and unexplored areas. There are links from room nodes to corresponding region and frontier nodes depicted which room they are located in. \n \
            At the lowest level 2 are object nodes and agent nodes. There is an edge from region node to each object node depicting which visited area of which room the object is located in. \
            There are also links between frontier nodes and objects nodes, depicting the objects in the vicinity of a frontier node. \n \
            Finally the agent node is where you are located in the environment. There is an edge between a region node and the agent node, depicting which visited area of which room the agent is located in."
        current_state_des = "'CURRENT STATE' will give you the exact location of the agent in the scene graph by giving you the agent node id, location, room_id and room name. "
        
        if self._use_image:
            current_state_des += " Additionally, you will also be given the current view of the agent as an image. "
        
        prompt = f'''You are an excellent hierarchical graph planning agent. 
            Your goal is to navigate an unseen environment to confidently answer a question about the environment.
            As you explore the environment, your sensors are building a scene graph representation (in json format) and you have access to that scene graph.  
            {scene_graph_desc}. {current_state_des} 
            Given the current state information, try to answer the question. Explain the reasoning for selecting the answer.
            The answer should be concise. Some examples of how the answers look like are:
            Question: Is the bed made? Answer: No.
            Question: I'm hot, what can I do? Answer: Turn on the dark grey fan.
            Question: Where is the black backpack? Answer: On the floor, next to the bed.
            Finally, report whether you are confident in answering the question. 
            Explain the reasoning behind the confidence level of your answer. Rate your level of confidence. 
            Provide a value between 0 and 1; 0 for not confident at all and 1 for absolutely certain.
            Do not use just commensense knowledge to decide confidence. 
            Choose TRUE, if you are certain about answering the question correctly given the scene graph and the image and no further exploration of the environment (which can provide you with a larger scene graph and more images) will help you answer the question better. 
            Choose 'FALSE', if you are uncertain of the answer and should explore the environment more by taking actions, and base your answer in the new observations. 
            Clarification: This is not your confidence in choosing the next action, but your confidence in answering the question correctly.
            If you are unable to answer the question with high confidence, and need more information to answer the question, then you can take two kinds of actions in the environment: 
            Goto_object_node_step or Goto_frontier_node_step. Choose the action that will enable you to answer the question better. 
            Goto_object_node_step Navigates the camera to near a certain object in the scene graph. Choose this action to get a good view of the region aroung this object, if you think going near this object will help you answer the question better.
            Important to note, the scene contains incomplete information about the environment (objects maybe missing, relationships might be unclear), so it is useful to go near relevant objects to get a better view to answer the question. 
            Use a scene graph as an imperfect guide to lead you to relevant regions to inspect. However, be extra careful and check the 'history' to check the previous actions you have taken.
            IMPORTANT: do not repeat past actions!! If you have already taken a action to goto a certain object, going to it again will give no new information, so avoid repeating Goto_object_node_step actions.
            Choose the object in a hierarchical manner by first reasoning about which room you should goto to best answer the question, and then choose the specific object. \n
            If the question explicitly asks to goto a different floor to answer the question try to look for stairs and go there.
            Choose Goto_frontier_node_step action if you think that using action "Goto_object_node_step" is not useful, in other words, if you think that going near any of the object nodes in the current scene graph will not provide you with any further information to answer the question better.
            This action will navigate you to a frontier (unexplored) region of the environment and will add breand new information about new objects/rooms to the scene graph. 
            If you are in the wrong room and a relevant room is not yet in the scene graph, choose this action since you need to explore unseen areas.
            Choose Goto_frontier_node_step based on the objects connected this frontier, in other words, Goto the frontier near which you see objects that are useful for answering the question or seem useful as a good exploration direction. Explain reasoning for choosing this frontier, by listing the list of objects (<id> and <name>) connected to this frontier node via a link (refer to scene graph) \n \
            
            Describe the CURRENT IMAGE. Pay special attention to features that can help answer the question or select future actions.
            Describe the SCENE GRAPH. Pay special attention to features that can help answer the question or select future actions.
            '''
        
        return prompt

    def get_current_state_prompt(self, scene_graph, agent_state):
        prompt = f"At t = {self.t}: \n \
            CURRENT AGENT STATE: {agent_state}. \n \
            SCENE GRAPH: {scene_graph}. \n  "
        
        if self._add_history:
            prompt += f"HISTORY: {self._history}"

        return prompt

    def update_history(self, agent_state, step, answer, target_pose):
        if step.__class__.__name__ == 'Goto_object_node_step':
            action = f"Goto object_id:{step.object_id.name} object name: {step.object_id.value}"
        else:
            action = f"Goto frontier_id:{step.frontier_id.name} frontier name: {step.frontier_id.value}"

        last_step = f'''
            [Agent state(t={self.t}): {agent_state}, 
            Action(t={self.t}): {action}, 
            Answer(t={self.t}): {answer.answer}
            Confidence(t={self.t}):  Confident: {answer.is_confident}, Confidence level:{answer.confidence_level}  \n
        '''
        self._history += last_step

    def evaluate_answer(self, question, answer, prediction, extra_answers):
        content = self.evaluation_prompt.format(
            question=question,
            answer=answer,
            prediction=prediction,
            extra_answers=extra_answers,
        )
        messages = [{"role": "user", "content": content}]

        completion = client.beta.chat.completions.parse(
            model=self._vlm_type,
            messages=messages
        )
        assert len(completion.choices) == 1
        output =  completion.choices[0].message.content

        tag = "Your mark:"
        if output.isdigit():
            score = int(output)
        else:
            start_idx = output.find(tag)
            if start_idx == -1:
                raise ValueError("Invalid output string: {}".format(output))
            else:
                end_idx = output.find("\n", start_idx)
                if end_idx == -1:
                    score = int(output[start_idx:].replace(tag, "").strip())
                else:
                    score = int(output[start_idx:end_idx].replace(tag, "").strip())
        succ = score > 3
        perc = 100.0 * (np.clip(score, 1, 5) - 1) / 4
        return succ, score, perc

    def get_gpt_output(self, current_state_prompt):
        
        messages=[
            {"role": "system", "content": f"AGENT ROLE: {self.agent_role_prompt}"},
            {"role": "system", "content": f"QUESTION: {self._question}"},
            {"role": "user", "content": f"CURRENT STATE: {current_state_prompt}."},
            # {"role": "user", "content": f"EXAMPLE PLAN: {self._example_plan}"} # TODO(saumya)
        ]

        if self._use_image:

            base64_image = encode_image(get_latest_image(self._output_path))
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

        frontier_node_list, room_node_list, region_node_list, object_node_list = self.get_actions()

        succ=False
        while not succ:
            try:
                start = time.time()
                completion = client.beta.chat.completions.parse(
                    model=self._vlm_type,
                    messages=messages,
                    response_format=create_planner_response(frontier_node_list, room_node_list, region_node_list, object_node_list, use_image=self._use_image),
                )
                plan = completion.choices[0].message
                if not (plan.refusal): # If the model refuses to respond, you will get a refusal message
                    succ=True
            except Exception as e:
                print(f"An error occurred: {e}. Sleeping for 60s")
                import ipdb; ipdb.set_trace()
                time.sleep(1)

        plan = completion.choices[0].message

        if len(plan.parsed.steps) > 0:
            step = plan.parsed.steps[0]
        else:
            step = None

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

        if step is None:
            return None, None, answer.is_confident, answer.confidence_level, answer.answer

        if step.__class__.__name__ == 'Goto_object_node_step':
            target_pose = self.sg_sim.get_position_from_id(step.object_id.name)
            target_id = step.object_id.name
            click.secho(f"Goto object node: {target_id} {step.object_id.value}",fg="green",)
        else:
            target_pose = self.sg_sim.get_position_from_id(step.frontier_id.name)
            target_id = step.frontier_id.name
            click.secho(f"Goto frontier node: {target_id}",fg="green",)

        if self._add_history:
            self.update_history(agent_state, step, answer, target_pose)

        self._t += 1
        return target_pose, target_id, answer.is_confident, answer.confidence_level, answer.answer
