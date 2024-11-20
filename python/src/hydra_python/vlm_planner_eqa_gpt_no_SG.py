import json
from enum import Enum
from typing import List, Tuple, Literal, Any, Union, Optional, Annotated
import time
import base64

from openai import OpenAI
from hydra_python.utils import get_instruction_from_eqa_data, get_latest_image
from pydantic import BaseModel

# client = OpenAI(
#     organization='org-9eg1dYLvm9Vnx13YZieDfE9n',
#     project='proj_rZU06lthKefMBx9rE3YGD2Um',
# )
client = OpenAI()

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def create_planner_response(Answer_options):
    
    class Answer(BaseModel):
        explanation_ans: str
        answer: Answer_options
        explanation_conf: str
        confidence_level: float
        is_confident: bool

    class PlannerResponse(BaseModel):
        answer: Answer
        image_description: str
    
    return PlannerResponse
    

class VLMPLannerEQAGPTNoSG:
    def __init__(self, cfg, question, pred_candidates, choices, answer, output_path):
        
        self._question, self.choices, self.vlm_pred_candidates = question, choices, pred_candidates
        self._answer = answer
        self._output_path = output_path
        self._vlm_type = cfg.name

        self.full_plan = ''
        self._t = 0

        self._outputs_to_save = [f'Question: {self._question}. \n Answer: {self._answer} \n']

    @property
    def t(self):
        return self._t
    
    def get_actions(self): 
        Answer_options = Enum('Answer_options', {token: choice for token, choice in zip(self.vlm_pred_candidates, self.choices)}, type=str)
        return Answer_options
    
    @property
    def agent_role_prompt(self):

        prompt = f'''
            You are an agent navigating an indoor environment and you are asked to answer a multiple choice question about the environment.
            Your goal is to confidently answer a multiple-choice question about the environment given some images.
            Explain the reasoning for selecting the answer.
            Report whether you are confident in answering the question. 
            Explain the reasoning behind the confidence level of your answer. Rate your level of confidence. 
            Provide a value between 0 and 1; 0 for not confident at all and 1 for absolutely certain.
            Do not use just commensense knowledge to decide confidence. 
            Choose TRUE, if you are certain about answering the question correctly and no further exploration will help you answer the question better. 
            Choose 'FALSE', if you are uncertain of the answer and should explore more to ground your answer in the current envioronment. 
            Clarification: This is not your confidence in choosing the next action, but your confidence in answering the question correctly.
            Describe the CURRENT IMAGE. Pay special attention to features that can help answer the question.
            '''
        return prompt

    def get_gpt_output(self):
        
        messages=[
            {"role": "system", "content": f"AGENT ROLE: {self.agent_role_prompt}"},
            {"role": "system", "content": f"QUESTION: {self._question}"},
        ]

        base64_image = encode_image(get_latest_image(self._output_path))
        messages.append(
            { 
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "CURRENT IMAGE: Use these images to answer the question."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            })

        Answer_options = self.get_actions()

        succ=False
        while not succ:
            try:
                start = time.time()
                completion = client.beta.chat.completions.parse(
                    model=self._vlm_type,
                    messages=messages,
                    response_format=create_planner_response(Answer_options),
                )
                plan = completion.choices[0].message
                if not (plan.refusal): # If the model refuses to respond, you will get a refusal message
                    succ=True
            except Exception as e:
                print(f"An error occurred: {e}. Sleeping for 60s")
                import ipdb; ipdb.set_trace()
                time.sleep(1)

        plan = completion.choices[0].message
        img_desc = plan.parsed.image_description
        return plan.parsed.answer, img_desc

    def get_next_action(self):

        answer, img_desc = self.get_gpt_output()

        # Saving outputs to file
        self._outputs_to_save.append(f'At t={self._t}:\
                                        Answer: {answer} \n \
                                        Image desc: {img_desc}  \n \n')
        self.full_plan = ' '.join(self._outputs_to_save)
        with open(self._output_path / "llm_outputs.json", "w") as text_file:
            text_file.write(self.full_plan)

        print(f'At t={self._t}: \n {answer}')

        self._t += 1
        return answer.is_confident, answer.confidence_level, answer.answer.name
