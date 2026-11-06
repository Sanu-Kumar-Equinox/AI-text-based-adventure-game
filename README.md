
# THE UNKNOWN WORLD

An AI-powered, text-based adventure game set in a mysterious post-apocalyptic world.

The game uses locally running AI models to generate interactive stories and illustrations.

## Features

- AI-generated stories
- Interactive player choices
- Locally running language model
- AI-generated scene illustrations
- Terminal-based interface

## AI Models

### Llama 3.2 3B Instruct
Used for generating stories and responding to player actions.

Download the GGUF model separately and place it inside the `AI` folder.

Expected filename:

`llama-3.2-3b-instruct-q4_k_m.gguf`

### Tiny-SD
Used for generating scene illustrations.

The model is downloaded automatically through Hugging Face when the game runs.

## Installation

1. Clone this repository.
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Create an `AI` folder.
4. Place the Llama GGUF model inside it.
5. Run the game:

```bash
python main.py
```

## Requirements

- Python
- Internet connection for initial model downloads
- Sufficient RAM and storage
- A compatible llama-cpp-python installation

## License and Attribution

See the NOTICE file for information about the third-party models and libraries used.
