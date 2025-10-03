from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from PIL import Image
import base64
import io
import os
import imghdr

# to avoid warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

# Configure Gemini model
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)

def process_image(image_path, prompt="Extract all text from this image"):
    try:
        # Read and encode the image
        with Image.open(image_path) as img:
            img_format = img.format
            if not img_format:
                detected_format = imghdr.what(image_path)
                img_format = detected_format.upper() if detected_format else "JPEG"
            
            # Convert to RGB if it's not already (for formats like PNG with transparency)
            if img.mode != "RGB" and img_format not in ["PNG", "GIF"]:
                img = img.convert("RGB")
            
            buffer = io.BytesIO()
            img.save(buffer, format=img_format)
            buffer.seek(0)
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Create message with image
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": f"data:image/{img_format.lower()};base64,{base64_image}"
                }
            ]
        )
        
        # Get response from LLM
        response = gemini_llm.invoke([message])
        return response.content
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None


def analyze_image(image_path, prompt=None):
    """
    Analyze an image with a specific prompt or auto-detect the best prompt
    """
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"
    
    # Default prompt if none provided
    if not prompt:
        prompt = "Extract all text from this image if there is any, otherwise describe what you see in detail."
    
    result = process_image(image_path, prompt)
    return result


if __name__ == "__main__":
    from rich.console import Console
    from langchain_core.messages import SystemMessage
    
    console = Console()
    
    # Default image path
    default_image_path = "f:/GitHub/Machine Learning/NLP, LLM and GenAI/LangChain/explore_myself/img2.jpg"
    

    # Initial system message
    system_message = SystemMessage(content="You're the best AI image assistant that can analyze any images or any format and generate good response by extracting text.")

    console.print("[bold blue]Image Analysis Chatbot[/bold blue]")
    console.print("[italic]N.B: Type 'exit', 'quit', or 'q' to end the conversation.[/italic]\n")

    while True:
        user_prompt = console.input("[bold red]You 👨🏻‍💻:[/bold red] ")

        # remove leading and trailing spaces
        user_prompt = user_prompt.strip()

        # If user is quitting
        if user_prompt.lower() in ("exit", "quit", "q"):
            console.print("[bold green]Goodbye! 👋[/bold green]")
            break

        # If user provided a prompt that is not empty
        if user_prompt:
            image_path = default_image_path

            # Analyze the image
            result = analyze_image(image_path, user_prompt)

            # Display result
            console.print(f"\n[bold yellow]AI 🤖:[/bold yellow] {result}")

        elif not user_prompt:
            console.print("[bold red]Error:[/bold red] Please provide a valid prompt or image path.")
        else:
            try:
                response = gemini_llm.invoke(user_prompt)
                console.print(f"\n[bold yellow]AI 🤖:[/bold yellow] {response.content}")
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")