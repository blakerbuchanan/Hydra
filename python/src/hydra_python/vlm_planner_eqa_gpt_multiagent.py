import json
from enum import Enum
from typing import List, Tuple, Literal, Any, Union, Optional, Annotated
import time
import base64

from openai import OpenAI
from hydra_python.utils import get_instruction_from_eqa_data
from pydantic import BaseModel

from vlm_planner_prompt import *

client = OpenAI()

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def gpt_completion_api(model: str, 
                     system_message:str, 
                     user_message:str, 
                     response_format, 
                     image_path=None):
    messages = [{"role": "system", "content": system_message}, 
               {"role": "user", "content": user_message}]
    if image_path:
        base64_image = encode_image(image_path)
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
    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages, 
        response_format = response_format
	)
    return response.choices[0].message

def create_planner_response(frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options):
    class Answer(BaseModel):
        explanation: str
        answer: Answer_options
        confidence: int
        image_description: str

    class Goto_frontier_node_step(BaseModel):
        frontier_id: frontier_node_list

    class Goto_object_node_step(BaseModel):
        object_id: object_node_list
    
    class SceneGraphResponse(BaseModel):
        node_id:  List[Union[Goto_object_node_step, Goto_frontier_node_step]]
        explanation: str
        scene_graph_description: str
    
    
    return SceneGraphResponse, Answer

class ReactResponse(BaseModel):
    thought: str
    action: str


class VLMPLannerEQAGPTMultiAgent:
    def __init__(self, cfg, sg_sim, question, pred_candidates, choices, answer, output_path):
        
        self._question, self.choices, self.vlm_pred_candidates = question, choices, pred_candidates
        self._answer = answer
        self._output_path = output_path
        self._vlm_type = cfg.name
        self._use_image = cfg.use_image

        self._example_plan = '' #TODO(saumya)
        self._history = []
        self.full_plan = ''
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
        Answer_options = Enum('Answer_options', {token: choice for token, choice in zip(self.vlm_pred_candidates+["NONE"], self.choices+["Not Sure."])}, type=str)
        return frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options


    def planner(self):
        carryover = ''
        for dct in self._history:
            carryover += f"THOUGHT: {dct['thought']}\n"
            carryover += f"ACTION: {dct['action']}\n"
            carryover += f"OBSERVATION: {dct['obs']}\n"

        ReAct_prompt = """
        Begin!
        QUESTION: {input}
        {carryover}
        """
        plan = gpt_completion_api(self._vlm_type, 
                                system_message= PLANNER_PROMPT + REACT_EXAMPLES,
                                user_message= ReAct_prompt.format(input=self._question, carryover=carryover),
                                response_format = ReactResponse
                                )
        return plan

    def scene_graph_planner(self, current_state_prompt, action, sg_response):
        out = gpt_completion_api(self._vlm_type, 
                                system_message= SG_PROMPT,
                                user_message= f"{current_state_prompt}TARGET: {action}",
                                response_format = sg_response
                                )
        return out

    def get_answer(self, answer_response):
        image_path = None
        if self._use_image:
            image_path = self._output_path / f"current_img_{self._t}.png"
        user_prompt = """"Use your current camera view to answer the following question: 
                           QUESTION: {input}"""
        
        answer = gpt_completion_api(self._vlm_type,
                                    system_message= ANSWER_PROMPT,          
                                    user_message= user_prompt.format(input=self._question),
                                    response_format = answer_response, 
                                    image_path = image_path
        )
        return answer

    def get_current_state_prompt(self, scene_graph, agent_state):
        prompt = f"At t = {self.t}: \n \
            CURRENT AGENT STATE: {agent_state}. \n \
            SCENE GRAPH: {scene_graph}. \n\n "
        return prompt


    def get_gpt_output(self, current_state_prompt):
        frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options = self.get_actions()
        sg_response, answer_response = create_planner_response(frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options)
        
        succ=False
        while not succ:
            try:
                start = time.time()
                
                ## High-level Planner
                react = self.planner()
                thought, action = react.parsed.thought, react.parsed.action
                
                ## Scene Graph Planner
                sg_output = self.scene_graph_planner(current_state_prompt, action, sg_response)
                steps, explanation, sg_desc = sg_output.parsed.node_id, sg_output.parsed.explanation, sg_output.parsed.scene_graph_description
                
                print(f"Time taken for planning next step: {time.time()-start}s")
                if not (sg_output.refusal): # If the model refuses to respond, you will get a refusal message
                    succ=True
                self._history.append({"thought": thought, "action": action})
            except Exception as e:
                print(f"An error occurred: {e}. Sleeping for 60s")
                breakpoint()
                time.sleep(1)

        if len(steps) > 0:
            step = steps[0]
        else:
            return None, None, None
        return step, answer_response, explanation, sg_desc


    def get_next_action(self):
        agent_state = self.sg_sim.get_current_semantic_state_str()
        current_state_prompt = self.get_current_state_prompt(self.sg_sim.scene_graph_str, agent_state)

        step, answer_response, node_explanation, sg_desc = self.get_gpt_output(current_state_prompt)
        
        if step is None:
            return None, None, False, 0, 0

        if step.__class__.__name__ == 'Goto_object_node_step':
            target_pose = self.sg_sim.get_position_from_id(step.object_id.name)
            target_id = step.object_id.name
        else:
            target_pose = self.sg_sim.get_position_from_id(step.frontier_id.name)
            target_id = step.frontier_id.name

        ## Answer
        answer = self.get_answer(answer_response)
        answer, confidence_level, explanation, img_desc = answer.parsed.answer, answer.parsed.confidence, answer.parsed.explanation, answer.parsed.image_description
        self._history[-1]["obs"] = f"*Current View:* {img_desc}"
        if len(self._history) > 1:
            self._history[-2]["obs"] += f" *Scene Graph:* {sg_desc}"
    
        # Saving outputs to file
        self._outputs_to_save.append(f'''At t={self._t}: 
                                        Agent state: {agent_state}
                                        Scene graph desc: {sg_desc}  
                                        Thought: {self._history[-1]["thought"]}
                                        Action: {self._history[-1]["action"]} 
                                        ------------------ 
                                        LLM output: {step}
                                        Explanation: {node_explanation}
                                        ----------------- 
                                        Image desc: {img_desc} 
                                        Answer: {answer}
                                        Confidence level: {confidence_level}
                                        Explanation: {explanation} \n'''
                                        )
        self.full_plan = ' '.join(self._outputs_to_save)
        with open(self._output_path / "llm_outputs.json", "w") as text_file:
            text_file.write(self.full_plan)

        print(f'At t={self._t}: \n {step} \n {answer}')

        self._t += 1
        return target_pose, (confidence_level>=4), confidence_level, answer.name
