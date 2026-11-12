"""Configuration constants for the harness."""

# Model routing: task_type -> model name
MODEL_ROUTING = {
    "coding": "qwen3.8-27b",
    "math": "qwen3.8-27b-noreason",
    "data_analysis": "qwen3.6-35b-a3b-noreason",
    "search": "qwen3.6-35b-a3b",
}

# Default LLM request parameters
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4096