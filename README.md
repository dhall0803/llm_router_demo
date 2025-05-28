# LLM Router Demo

A toy Python application for learning purposes that intelligently routes user questions to the most appropriate Large Language Model (LLM) based on the complexity of the query.

## Overview

The LLM Router Demo automatically analyzes user questions and selects the most suitable model from a range of available options, balancing performance requirements with cost considerations. The application uses a routing model to determine which LLM would best answer the user's question.

This project was created for educational purposes to demonstrate:
- Intelligent routing between multiple LLM models
- API integration with different AI providers
- Parameter validation and error handling
- Command-line interface design

## Available Models

- **gpt-4.1-nano**: For simple tasks requiring quick responses at low cost
- **gpt-4.1**: For complex tasks with balanced performance and cost
- **deepseek-reasoner**: For very complex tasks requiring highest performance (at higher cost)

## Features

- **Intelligent Model Selection**: Automatically selects the most appropriate model based on question complexity
- **Input Validation**: Ensures API responses are properly formatted and valid
- **Cost Optimization**: Matches query complexity to the most cost-effective model capable of answering it
- **Comprehensive Testing**: Includes unit tests and end-to-end tests for reliability

## Requirements

- Python 3.12+
- OpenAI API Key (for GPT-4.1-nano and GPT-4.1)
- Deepseek API Key (for deepseek-reasoner)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd llm_router_demo
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   # or
   uv pip install -e .
   ```

4. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   DEEPSEEK_API_KEY=your_deepseek_api_key
   ```

## Usage

Run the program:
```bash
python main.py
```

You will be prompted to enter a question, and the system will:
1. Analyze your question to determine its complexity
2. Select the most appropriate model
3. Display the reasoning for the model selection
4. Fetch and display the response from the selected model

## How It Works

1. User input is received through the command line interface
2. The application uses a lightweight model (gpt-4.1-nano) to analyze the question complexity
3. Based on the analysis, it routes the question to the most suitable model
4. The chosen model processes the question and returns a response
5. The response is displayed to the user along with information about which model was used

## Testing

The project includes several test modules:
- `test_basic.py`: Basic functionality tests
- `test_end_to_end.py`: End-to-end testing
- `test_model_selection.py`: Tests for the model selection logic
- `test_validation.py`: Input validation tests

Run tests using pytest:
```bash
pytest
```

## Project Structure

- `main.py`: Main application code
- `pyproject.toml`: Project dependencies and metadata
- `.env`: Environment variables (API keys)
- `tests/`: Test modules

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.