import os
import logging
import warnings

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import torch
from llama_cpp import Llama
from diffusers import DiffusionPipeline
from diffusers.utils import logging as diffusers_logging
from huggingface_hub.utils import disable_progress_bars

from rich.console import Console
from rich.panel import Panel


diffusers_logging.disable_progress_bar()
disable_progress_bars()

console = Console()

console.clear()

with console.status("[cyan]Preparing your adventure...[/cyan]"):

    llm = Llama(
        model_path="./AI/llama-3.2-3b-instruct-q4_k_m.gguf",
        n_ctx=4096,
        verbose=False
    )

image_ai = None


def load_image_ai():

    global image_ai

    if image_ai is None:

        with console.status("[cyan]Preparing image engine...[/cyan]"):

            image_ai = DiffusionPipeline.from_pretrained(
                "segmind/tiny-sd",
                torch_dtype=torch.float32,
                use_safetensors=False,
                revision="main"
            )

            image_ai.to("cpu")
            image_ai.enable_attention_slicing()
            image_ai.set_progress_bar_config(disable=True)


def show_image(image):

    image = image.convert("RGB")
    image.thumbnail((125, 125))

    width, height = image.size
    pixels = image.load()

    print()

    for y in range(0, height, 2):

        line = ""

        for x in range(width):

            top = pixels[x, y]

            if y + 1 < height:
                bottom = pixels[x, y + 1]
            else:
                bottom = top

            line += (
                f"\033[38;2;{top[0]};{top[1]};{top[2]}m"
                f"\033[48;2;{bottom[0]};{bottom[1]};{bottom[2]}m"
                "▀"
            )

        print(line + "\033[0m")

    print()


def make_image(scene):

    load_image_ai()

    # Remove the question from the image description
    image_prompt = scene.rsplit("What do you do?", 1)[0]

    with console.status("[cyan]Painting the scene...[/cyan]"):

        picture = image_ai(
            prompt=(
                image_prompt
            ),
            num_inference_steps=6,
            height=480,
            width=480
        ).images[0]

    os.makedirs("generated_images", exist_ok=True)

    picture.save("generated_images/scene.png")

    show_image(picture)



messages = [

    {
        "role": "system",
        "content": """
       You are the master and manager of a text based story game.
       Generate a story that is interestint to play through.
       Each round give the player a choice about what they want to do.
       If the player is not realistic about ther actions, give them a warning.

       RULES:
       - Make sure to keep the resonse around  50 to 90 words.
       - Make sure to use simple language.
       - Have a definite ending revealed to the player as they progress.
       - The player can do whatever they want but they can only countroll the action of ther body.
       - The player cannot just change the enviornment however they want, for example they cannot just say "a plane lands near me" or "I see this".
       - NEVER let the player decide what they can see or hear, they can only choose what to do next.
        """
    },

    {
        "role": "user",
        "content": "Begin the adventure. Create an intriguing opening scene."
    }

]


def show_title():

    console.print(
        Panel(
            "[bold cyan]THE UNKNOWN WORLD[/bold cyan]\n"
            "[dim]A mysterious AI-powered adventure[/dim]",
            border_style="cyan"
        )
    )


while True:

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=500,
        temperature=0.8
    )

    scene = response["choices"][0]["message"]["content"]

    messages.append({
        "role": "assistant",
        "content": scene
    })

    console.clear()

    show_title()

    console.print(
        Panel(
            scene,
            title="[bold]CURRENT SCENE[/bold]",
            border_style="cyan"
        )
    )

    try:

        make_image(scene)

    except Exception:

        console.print(
            "[dim]The scene illustration is unavailable.[/dim]"
        )

    action = console.input(
        "\n[bold green]Your action > [/bold green]"
    )

    if action.lower().strip() in ["quit", "exit"]:

        console.clear()

        console.print(
            Panel(
                "[yellow]Your adventure has ended.[/yellow]",
                border_style="yellow"
            )
        )

        break

    messages.append({
        "role": "user",
        "content": action
    })
