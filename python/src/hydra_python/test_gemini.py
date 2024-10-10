import google.generativeai as genai
import os
import PIL.Image
from markdown import Markdown
import enum
from typing_extensions import TypedDict

class Choice(enum.Enum):
    A = "Answer: A"
    B = "Answer: B"
    C = "Answer: C"
    D = "Answer: D"

class Response(TypedDict):
    answer_text: str
    answer = Choice

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

sample_file = PIL.Image.open('/home/saumyas/catkin_ws_semnav/data/gemini_test/bedside_lamp_off.jpg')

# Choose a Gemini model.
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

prompt = "What is the status of the lamp next to the bed on top of the bedside table? A) On B) Off C) Do not use this option D) Do not use this option. "

response = model.generate_content([prompt, sample_file],
            generation_config=genai.GenerationConfig(
            response_mime_type="text/x.enum", response_schema=Response),
        )

print(response.text)

boots_image = PIL.Image.open('/home/saumyas/catkin_ws_semnav/data/gemini_test/boots_n_shoes.jpg')
prompt = "How many pairs of long boots are in the image? Be sure to distinguish between boots and shoes. A) Three pairs B) Two pairs C) Four pairs D) One pair."

response = model.generate_content([prompt, boots_image],
            generation_config=genai.GenerationConfig(
            response_mime_type="text/x.enum", response_schema=Response),
        )

print(response.text)

prompt = "How many pairs of footwear are in the image? A) Three pairs B) Two pairs C) Four pairs D) One pair."

response = model.generate_content([prompt, boots_image],
            generation_config=genai.GenerationConfig(
            response_mime_type="text/x.enum", response_schema=Response),
        )

print(response)