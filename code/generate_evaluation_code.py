import json
from vLLM_Engine import vLLM_Engine
import os

Prompt = """
Generate Python template `evaluate` functions to validate a response against the given constraint type. Follow these rules:

1. Core Function:
   - Must have `response` parameter
   - Can add other necessary parameters as {{template_variables}}
   - Return boolean value

2. Smart Extension:
   - Analyze the constraint type and create logical variants
   - Try to include multiple extensions where possible, but always ensure accurate evaluation.
   - Example extensions:
     * For length constraints: per-paragraph/per-section checks
     * For linguistic features: per-sentence/per-word checks
     * For structural rules: nested element validation
   - Each variant should serve a distinct validation purpose
   - Note that all extensions must be correctly evaluated by the evaluation code
   - If extending to paragraph-level validation, the code should explicitly specify how paragraphs are separated, such as by two newline characters.

3. Output Requirements:
   - Only include Python code in ```python``` blocks
   - Each function in separate code blocks
   - Include clear docstring explaining the variant

Example for "maximum words":
```python
def evaluate_max_words(response, {{max_words}}):
    ## Validate total word count <= {{max_words}}.
    from nltk.tokenize import word_tokenize
    words = word_tokenize(response)
    return len(words) <= {{max_words}}
```
```python
def evaluate_max_words_per_sentence(response, {{max_per_sentence}}):
    ## Ensure no sentence exceeds {{max_per_sentence}} words.
    from nltk.tokenize import sent_tokenize, word_tokenize
    sentences = sent_tokenize(response)
    for sent in sentences:
        if len(word_tokenize(sent)) > {{max_per_sentence}}:
            return False
    return True
```
Now generate appropriate evaluation functions for this constraint:
{constraint}
"""

def load_constraint_type(constraint_type_file_path, lang="en"):
   leaf_elements = []
   with open(os.path.join(constraint_type_file_path, f"constraint_type_{lang}.json"), "r") as f:
      data = json.load(f)
   for item in data:
      for category, subitems in item.items():
         for subitem in subitems:
               for key, values in subitem.items():
                  if isinstance(values, list):
                     leaf_elements.extend(values)
      
   return leaf_elements

def extract_code_blocks(response):
    pattern = r'```python(.*?)```'
    code_blocks = re.findall(pattern, text, re.DOTALL)
    cleaned_blocks = [block.strip() for block in code_blocks]
    
    return cleaned_blocks


if __name__ == "__main__":
   constraint_type_file_path = "data"
   model_name_or_path = "../models/qwen3-32b"
   decoding_dict = {
      "max_tokens": 32768,
      "temperature": 0.6,
      "top_p": 0.95,
      "top_k": 20
   }
   tp = 4

   constraints = load_constraint_type(constraint_type_file_path)
   vllm_engine = vLLM_Engine(
      model_name_or_path=model_name_or_path,
      decoding_dict=decoding_dict,
      tp=tp,
   )
   prompts = []
   for c in constraints:
      prompts.append(
         {
            "user": Prompt.format(c)
         }
      )
   responses = vllm_engine.generate(prompts)

   evaluation_codes = []
   for resp in responses:
      evaluation_codes.extend(extract_code_blocks(resp))