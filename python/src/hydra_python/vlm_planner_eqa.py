import json
from enum import Enum
from typing import Union
from typing import List, Tuple
import time
import hydra_python as hydra

from openai import OpenAI

client = OpenAI()

from pydantic import BaseModel


def create_planner_response(Goto_visited_node_action, Goto_frontier_node_action, Answer_options):
    class Done_action(str, Enum):
        Done = "done_with_task"
    
    class Confidence_check(str, Enum):
        Confident_in_answering_question = "yes"
        Not_confident_in_answering_question = "no"

    class Goto_visited_node_step(BaseModel):
        explanation_visited: str
        action: Goto_visited_node_action

    class Goto_frontier_node_step(BaseModel):
        explanation_frontier: str
        action: Goto_frontier_node_action

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
        steps: List[Union[Goto_visited_node_step, Goto_frontier_node_step, Done_step]]
        answer: Answer
        confidence: Confidence
        # final_full_plan: str
    
    return PlannerResponse

class VLMPLannerEQA:
    def __init__(self, question_data, output_path, pipeline, rr_logger, frontier_nodes=None):
        
        self._question = self._get_instruction(question_data)
        self._answer = question_data["answer"]
        self._output_path = output_path

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
        Goto_frontier_node_action = Enum('Goto_frontier_node_action', {ac: ac for ac in self.sg_sim.frontier_node_ids}, type=str)
        Answer_options = Enum('Answer_options', {token: choice for token, choice in zip(self.vlm_pred_candidates, self.choices)}, type=str)
        return Goto_visited_node_action, Goto_frontier_node_action, Answer_options
    
    @property
    def agent_role_prompt(self):
        prompt = "You are an excellent graph planning agent. \
            You are given a scene graph representation (in json format) of the areas of the environment you have explored so far. \
            Nodes in the scene graph will give you information about the 'buildings', 'rooms', 'visited' nodes, 'frontier' nodes and 'objects' in the scene.\
            Edges in the scene graph tell you about connected components in the scenes: For example, Edge from a room node to object node will tell you which objects are in which room.\
            Frontier nodes and visited nodes are empty spaces in the scene where the agent can navigate to.\
            Frontier nodes represent areas that are at the boundary of visited and unexplored empty areas.\
            Edges among frontier nodes and visited nodes tell which empty spaces are connected to each other, hence can be reached from one another.\
            You are tasked with 'exploring' a previously unseen enviornment to Answer a multiple-choice question about the environment. Keep exploring until you can confidently answer the question. \
            You are also required to report whether using the scene graph and your current state, you are able to answer the question with high Confidence.\
            Finally, You can take three kinds of steps in the environment: Goto_visited_node_step, Goto_frontier_node_step and Done_step \n \
            1) Goto_visited_node_step: Navigate to a visited node. Scene graph may or may not be augmented depending upon how well the region was explored when visited before \n \
            2) Goto_frontier_node_step: Navigate to a frontier (unexplored) node. Going to frontier node will proide the agent with new observations and the scene graph will be augmented. \n \
            3) Done_step: Check the current state and scene graph carefully. If the question can be answered with high confidence, then only take the done action else take one of the other actions. \n "
        return prompt
    
    def get_current_state_prompt(self, scene_graph, agent_state):
        # TODO(saumya): Include history
        prompt = f"At t = {self.t}: \n \
            CURRENT AGENT STATE: {agent_state}. \n \
            SCENE GRAPH: {scene_graph}. \n "
        return prompt

    def get_next_action(self):
        # self.sg_sim.update()
        
        agent_state = self.sg_sim.get_current_semantic_state_str()

        current_state_prompt = self.get_current_state_prompt(self.sg_sim.scene_graph_str, agent_state)

        messages=[
            {"role": "system", "content": f"AGENT ROLE: {self.agent_role_prompt}"},
            {"role": "system", "content": f"QUESTION: {self._question}"},
            {"role": "user", "content": f"CURRENT STATE: {current_state_prompt}."},
            # {"role": "user", "content": f"EXAMPLE PLAN: {self._example_plan}"} # TODO(saumya)
        ]
        Goto_visited_node_action, Goto_frontier_node_action, Answer_options = self.get_actions()

        succ=False
        while not succ:
            try:
                start = time.time()
                completion = client.beta.chat.completions.parse(
                    # model="gpt-4o-mini",
                    model="gpt-4o-2024-08-06",
                    messages=messages,
                    response_format=create_planner_response(Goto_visited_node_action, Goto_frontier_node_action, Answer_options),
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
        print(f'At t={self._t}: \n {step}')

        if 'done_with_task' in step.action.value:
            self._done = True
            target_pose = None
        else:
            target_pose = self.sg_sim.get_position_from_id(step.action.value)

        # Saving outputs to file
        self._outputs_to_save.append(f'At t={self._t}: \n \
                                        Agent state: {agent_state} \n \
                                        LLM output: {step}. \n \
                                        Confidence: {plan.parsed.confidence} \n \
                                        Answer: {plan.parsed.answer} \n \n')
        self.full_plan = ' '.join(self._outputs_to_save)
        with open(self._output_path / "llm_outputs.json", "w") as text_file:
            text_file.write(self.full_plan)

        self._t += 1
        return target_pose, self.done, plan.parsed.confidence.answer.value, plan.parsed.answer.answer.name
        
