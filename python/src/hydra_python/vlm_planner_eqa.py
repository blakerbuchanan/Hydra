import json
from enum import Enum
from typing import List, Tuple, Literal, Any, Union, Optional
import time
import hydra_python as hydra
import base64

from openai import OpenAI
import google.generativeai as genai
import os

# client = OpenAI(
#     organization='org-9eg1dYLvm9Vnx13YZieDfE9n',
#     project='proj_rZU06lthKefMBx9rE3YGD2Um',
# )
client = OpenAI()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Choose a Gemini model.
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

from pydantic import BaseModel

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def create_planner_response(Goto_visited_node_action, Goto_object_node_action, Goto_frontier_node_action, Answer_options):
    class Done_action(str, Enum):
        Done = "done_with_task"
    
    class Confidence_check(str, Enum):
        Confident_in_correctly_answering_question = "yes"
        Not_confident_in_correctly_answering_question = "no"

    class Goto_visited_node_step(BaseModel):
        explanation_visited: str
        action: Goto_visited_node_action

    class Goto_frontier_node_step(BaseModel):
        explanation_frontier: str
        action: Goto_frontier_node_action

    class Goto_object_node_step(BaseModel):
        explanation_obj: str
        action: Goto_object_node_action

    class Done_step(BaseModel):
        explanation_done: str
        action: Done_action
    
    class Answer(BaseModel):
        explanation_ans: str
        answer: Answer_options

    class Confidence(BaseModel):
        explanation_conf: str
        answer: Confidence_check

    class PlannerResponse(BaseModel):
        steps: List[Union[Goto_object_node_step, Goto_frontier_node_step]]
        answer: Answer
        confidence: Confidence
        image_description: str
    
    return PlannerResponse
    
def create_planner_response_schema(Goto_visited_node_action, Goto_object_node_action, Goto_frontier_node_action, Answer_options):

    class Done_action(str, Enum):
        Done = "done_with_task"

    # Combine all action types into a single list
    all_actions = list(Goto_object_node_action) + list(Goto_frontier_node_action) + list(Done_action)

    return {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["object_node", "frontier_node", "done_step"]
                        },
                        "explanation_obj": {"type": "string"},
                        "explanation_frontier": {"type": "string"},
                        "explanation_done": {"type": "string"},
                        "action": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "enum": all_actions},
                                "value": {"type": "string"}
                            },
                            "required": ["name", "value"]
                        }
                    },
                    "required": ["type", "action"]
                }
            },
            "answer": {
                "type": "object",
                "properties": {
                    "explanation_ans": {"type": "string"},
                    "answer": {"type": "string", "enum": list(Answer_options)}
                },
                "required": ["explanation_ans", "answer"]
            },
            "confidence": {
                "type": "object",
                "properties": {
                    "explanation_conf": {"type": "string"},
                    "answer": {"type": "string", "enum": ["yes", "no"]}
                },
                "required": ["explanation_conf", "answer"]
            }
        },
        "required": ["steps", "answer", "confidence"]
    }

class VLMPLannerEQA:
    def __init__(self, cfg, question_data, output_path, pipeline, rr_logger, frontier_nodes=None):
        
        self._question = self._get_instruction(question_data)
        self._answer = question_data["answer"]
        self._output_path = output_path
        self._vlm_type = cfg.name
        self._use_image = cfg.use_image

        self._example_plan = '' #TODO(saumya)
        self._done = False
        self._history = ''
        self._t = 0

        self._outputs_to_save = [f'Question: {self._question}. \n Answer: {self._answer} \n']

        self.sg_sim = hydra.SceneGraphSim(output_path, pipeline, rr_logger, frontier_nodes)

    def _get_instruction(self, question_data):
        question = question_data["question"]
        self.choices = [c.split("'")[1] for c in question_data["choices"].split("',")]
        # Re-format the question to follow LLaMA style
        vlm_question = question
        self.vlm_pred_candidates = ["A", "B", "C", "D"]
        for token, choice in zip(self.vlm_pred_candidates, self.choices):
            vlm_question += "\n" + token + "." + " " + choice
        
        return vlm_question

    @property
    def done(self):
        return self._done
    
    @property
    def t(self):
        return self._t
    
    def get_actions(self):
        Goto_visited_node_action = Enum('Goto_visited_node_action', {ac: ac for ac in self.sg_sim.visited_node_ids}, type=str)
        Goto_object_node_action = Enum('Goto_object_node_action', {id: name for id, name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names)}, type=str)
        Goto_frontier_node_action = Enum('Goto_frontier_node_action', {ac: ac for ac in self.sg_sim.frontier_node_ids}, type=str)
        Answer_options = Enum('Answer_options', {token: choice for token, choice in zip(self.vlm_pred_candidates, self.choices)}, type=str)
        return Goto_visited_node_action, Goto_object_node_action, Goto_frontier_node_action, Answer_options
    
    # @property
    # def agent_role_prompt(self):
    #     prompt = "You are an excellent graph planning agent. \
    #         You are given a scene graph representation (in json format) of the areas of the environment you have explored so far. \
    #         Nodes in the scene graph will give you information about the 'buildings', 'rooms', 'visited' nodes, 'frontier' nodes and 'objects' in the scene.\
    #         Edges in the scene graph tell you about connected components in the scenes: For example, Edge from a room node to object node will tell you which objects are in which room.\
    #         Frontier nodes and visited nodes are empty spaces in the scene where the agent can navigate to.\
    #         Frontier nodes represent areas that are at the boundary of visited and unexplored empty areas.\
    #         Edges among frontier nodes and visited nodes tell which empty spaces are connected to each other, hence can be reached from one another.\
    #         You are tasked with 'exploring' a previously unseen enviornment to Answer a multiple-choice question about the environment. Keep exploring until you can confidently answer the question. \
    #         You are also required to report whether using the scene graph and your current state, you are able to answer the question with high Confidence.\
    #         Finally, You can take three kinds of steps in the environment: Goto_visited_node_step, Goto_frontier_node_step and Done_step \n \
    #         1) Goto_visited_node_step: Navigate to a visited node. Scene graph may or may not be augmented depending upon how well the region was explored when visited before \n \
    #         2) Goto_frontier_node_step: Navigate to a frontier (unexplored) node. Going to frontier node will proide the agent with new observations and the scene graph will be augmented. \n \
    #         3) Done_step: Check the current state and scene graph carefully. If the question can be answered with high confidence, then only take the done action else take one of the other actions. \n "
    #     return prompt
    
    
    @property
    def agent_role_prompt(self):
        prompt = "You are an excellent graph planning agent. Your goal is to explore an environment to multiple-choice question about the environment.\
            As you explore the environment, your sensors are building a scene graph representation (in json format) and you have access to that scene graph.  \
            Nodes in the scene graph will give you information about the 'buildings', 'rooms', 'visited' nodes, 'frontier' nodes and 'objects' in the scene.\
            Edges in the scene graph tell you about connected components in the scenes: For example, Edge from a room node to object node will tell you which objects are in which room.\
            Frontier nodes represent areas that are at the boundary of visited and unexplored empty areas. Edges from frontiers to objects denote which objects are close to that frontier node. Use this information to choose the next frontier to explore.\
            You are required to report an answer to the question, even if Confidence is low. This answer should include the letter associated with the choices available.\
            You are required to report whether using the scene graph and your current state and image, you are able to answer the question 'CORRECTLY' with high Confidence.\
            You are also required to provide a brief description of the current image 'image_description' you are given and explain if that image has any useful features that can help answer the question. \n \
            To explore the environment choose between two actions: Goto_frontier_node_step and Goto_object_node_step. \n \
            Goto_frontier_node_step: Navigates to a frontier (unexplored) node and will provide the agent with new observations and the scene graph will be augmented. \n \
            Goto_object_node_step: Navigates to a certain seen object. This can help facilitate going back to a previously explored area to answer the question related to an object in the question."
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

        Goto_visited_node_action, Goto_object_node_action, Goto_frontier_node_action, Answer_options = self.get_actions()

        succ=False
        while not succ:
            try:
                start = time.time()
                completion = client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    # model="gpt-4o-2024-08-06",
                    messages=messages,
                    response_format=create_planner_response(Goto_visited_node_action, Goto_object_node_action, Goto_frontier_node_action, Answer_options),
                )
                print(f"Time taken for planning next step: {time.time()-start}s")
                plan = completion.choices[0].message
                if not (plan.refusal): # If the model refuses to respond, you will get a refusal message
                    succ=True
            except Exception as e:
                print(f"An error occurred: {e}. Sleeping for 60s")
                time.sleep(1)

        plan = completion.choices[0].message
        step = plan.parsed.steps[0]

        if self._use_image:
            img_desc = plan.parsed.image_description
        else:
            img_desc = ' '
        
        return step, plan.parsed.confidence, plan.parsed.answer, img_desc
        
    def get_gemini_output(self, current_state_prompt):
        # TODO(blake):
        messages=[
            {"role": "model", "parts": [{"text": f"AGENT ROLE: {self.agent_role_prompt}"}]},
            {"role": "model", "parts": [{"text": f"QUESTION: {self._question}"}]},
            {"role": "user", "parts": [{"text": f"CURRENT STATE: {current_state_prompt}."}]},
            # {"role": "user", "content": f"EXAMPLE PLAN: {self._example_plan}"} # TODO(saumya)
        ]

        Goto_visited_node_action, Goto_object_node_action, Goto_frontier_node_action, Answer_options = self.get_actions()

        succ=False
        while not succ:
            try:
                start = time.time()

                response = gemini_model.generate_content(messages,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json", response_schema=create_planner_response_schema(Goto_visited_node_action, Goto_object_node_action, Goto_frontier_node_action, Answer_options)),
                )

                print(f"Time taken for planning next step: {time.time()-start}s")
                if (True): # If the model refuses to respond, you will get a refusal message
                    succ=True
            except Exception as e:
                print(f"An error occurred: {e}. Sleeping for 45s")
                time.sleep(45)
        
        json_response = response.text
        response_dict = json.loads(json_response)

        step = response_dict["steps"][0]
        confidence = response_dict["confidence"]
        answer = response_dict["answer"]["answer"]

        return step, confidence, answer
    

    def get_next_action(self):
        # self.sg_sim.update()
        
        agent_state = self.sg_sim.get_current_semantic_state_str()

        current_state_prompt = self.get_current_state_prompt(self.sg_sim.scene_graph_str, agent_state)

        if self._vlm_type == 'gpt':
            step, confidence, answer, img_desc = self.get_gpt_output(current_state_prompt)

        if self._vlm_type == 'gemini':
            step, confidence, answer, img_desc = self.get_gemini_output(current_state_prompt)


        print(f'At t={self._t}: \n {step}')
        if self._vlm_type == 'gemini':
            if 'done_with_task' in step["action"]["name"]:
                self._done = True
                target_pose = None
            else:
                target_pose = self.sg_sim.get_position_from_id(step["action"]["name"])
        if self._vlm_type == 'gpt':
            if 'done_with_task' in step.action.value:
                self._done = True
                target_pose = None
            else:
                target_pose = self.sg_sim.get_position_from_id(step.action.name)

        # Saving outputs to file
        self._outputs_to_save.append(f'At t={self._t}: \n \
                                        Agent state: {agent_state} \n \
                                        LLM output: {step}. \n \
                                        Confidence: {confidence} \n \
                                        Answer: {answer} \n \
                                        Image desc: {img_desc} \n \n')
        self.full_plan = ' '.join(self._outputs_to_save)
        with open(self._output_path / "llm_outputs.json", "w") as text_file:
            text_file.write(self.full_plan)

        self._t += 1
        if self._vlm_type == 'gpt':
            return target_pose, self.done, confidence.answer.value, answer.answer.name
        if self._vlm_type == 'gemini':
            return target_pose, self.done, confidence["answer"], answer
        
