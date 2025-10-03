from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from rich.console import Console
import os
import warnings

# Avoid warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize console and load environment variables
console = Console()
load_dotenv()

# Configure Gemini model
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)

def check_grammar(text):
    """
    Check and correct grammar in the provided text
    """
    try:
        # Create message with specific grammar checking instructions
        message = HumanMessage(
            content=f"Check the following text for grammar errors and correct them. Return only the corrected text without explanations: \n\n{text}"
        )
        
        # Get response from LLM
        response = gemini_llm.invoke([message])
        return response.content
        
    except Exception as e:
        console.print(f"[bold red]Error checking grammar: {str(e)}[/bold red]")
        return None

def paraphrase_text(text, style="standard"):
    """
    Paraphrase the provided text in the specified style
    
    Styles:
    - standard: Normal paraphrasing
    - formal: More professional language
    - simple: Easier to understand
    - creative: More engaging and unique
    """
    try:
        # Create message with paraphrasing instructions
        style_instruction = ""
        if style == "formal":
            style_instruction = "Use formal, professional language."
        elif style == "simple":
            style_instruction = "Use simple, easy-to-understand language."
        elif style == "creative":
            style_instruction = "Be creative and engaging in your paraphrasing."
        
        message = HumanMessage(
            content=f"Paraphrase the following text. {style_instruction} Return only the paraphrased text without explanations: \n\n{text}"
        )
        
        # Get response from LLM
        response = gemini_llm.invoke([message])
        return response.content
        
    except Exception as e:
        console.print(f"[bold red]Error paraphrasing text: {str(e)}[/bold red]")
        return None

def summarize_text(text, length="medium"):
    """
    Summarize the provided text
    
    Length options:
    - short: 1-2 sentences
    - medium: 3-5 sentences
    - long: 6-8 sentences
    """
    try:
        # Determine summary length
        length_instruction = ""
        if length == "short":
            length_instruction = "Keep the summary to 1-2 sentences."
        elif length == "medium":
            length_instruction = "Keep the summary to 3-5 sentences."
        elif length == "long":
            length_instruction = "Keep the summary to 6-8 sentences."
        
        # Create message with summarization instructions
        message = HumanMessage(
            content=f"Summarize the following text. {length_instruction} Return only the summary without explanations: \n\n{text}"
        )
        
        # Get response from LLM
        response = gemini_llm.invoke([message])
        return response.content
        
    except Exception as e:
        console.print(f"[bold red]Error summarizing text: {str(e)}[/bold red]")
        return None

# Example usage
if __name__ == "__main__":
    console.print("[bold blue]Text Processing Tool[/bold blue]")
    
    while True:
        user_prompt = console.input("[bold red]You 👨🏻‍💻:[/bold red] ")
        
        # If user is quitting
        if user_prompt.lower() in ("exit", "quit", "q"):
            console.print("[bold green]Goodbye! 👋[/bold green]")
            break
        
        # Process based on command
        if user_prompt.lower().startswith("grammar:"):
            text = user_prompt[8:].strip()
            result = check_grammar(text)
            console.print(f"\n[bold yellow]AI 🤖:[/bold yellow] {result}")
            
        elif user_prompt.lower().startswith("paraphrase:"):
            # Check for style specification
            parts = user_prompt[11:].strip().split(" style:", 1)
            text = parts[0].strip()
            style = "standard"
            if len(parts) > 1:
                style = parts[1].strip().lower()
            
            result = paraphrase_text(text, style)
            console.print(f"\n[bold yellow]AI 🤖:[/bold yellow] {result}")
            
        elif user_prompt.lower().startswith("summarize:"):
            # Check for length specification
            parts = user_prompt[10:].strip().split(" length:", 1)
            text = parts[0].strip()
            length = "medium"
            if len(parts) > 1:
                length = parts[1].strip().lower()
            
            result = summarize_text(text, length)
            console.print(f"\n[bold yellow]AI 🤖:[/bold yellow] {result}")
            
        else:
            console.print("\n[bold yellow]AI 🤖:[/bold yellow] Please use one of the following commands:")
            console.print("  - grammar: [your text]")
            console.print("  - paraphrase: [your text] style:[standard|formal|simple|creative]")
            console.print("  - summarize: [your text] length:[short|medium|long]")
            console.print("  - exit, quit, or q to end the conversation")