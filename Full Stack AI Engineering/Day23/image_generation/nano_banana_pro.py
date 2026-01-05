from google import genai
from PIL import Image
import os

import time
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

prompt = f"""
Create a manga of these characters and show the stand off between Messi and Ronaldo.
A high-impact, comic-style mega-mashup of iconic meme figures collaboratively driving an action-packed, retro-vibe crossover scene.
A dynamic and intense close-up portrait in a graphic novel or comic book art style, featuring a male character with long, dark, slightly messy hair and a short beard, wearing a suit and tie. His eyes are glowing with furious, stylized yellow energy or smoke, extending outwards. His expression is one of extreme anger, determination, and pain, with gritted teeth and visible facial scars or distress. Dominating the image is a bold, bright red "X" drawn over his face, extending from corner to corner of the frame, symbolizing danger, a target, or a "no entry" warning. The artwork is characterized by strong, confident black ink lines, high contrast, and a gritty, textured, almost sketched quality. The color palette is stark: black, white, red, and vibrant yellow for the eyes, on a clean white or light grey background.
"""

start_time = time.monotonic()
response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=prompt,
)

for part in response.parts:
    if part.inline_data:
        image = part.as_image()
        image.show()
        timestamp = int(time.time())
        filename = f"generated_image_{timestamp}.png"
        image.save(filename)
        print(f"Image saved successfully as '{filename}'")
end_time = time.monotonic()

print(f"Time taken to generate the image: {end_time - start_time} seconds")
