# Evidence level grader using LLMs and Machine Learning

# Project structure
Here is a brief overview of the project structure:

- data/: used to store raw and preprocessed CIViC data (.csv)
- src/: scripts for data spliting and preprocessing, machine learning models training, i.e., xgboost and decision tree, and LLMs inference for classification of evidence level
- runs/: used to store output data from src/ (e.g., dataframes and visualizations from models performance evaluation)

# Installation instructions
Create a .env file with your OpenAI and Gemini API keys following this format:

``` 
OPENAI_API_KEY='yourkey'
GEMINI_API_KEY='yourkey'
```



