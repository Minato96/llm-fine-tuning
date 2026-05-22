from datasets import load_dataset
import pandas as pd
import os

class dataset_handler:
    def __init__(self,path="code-search-net/code_search_net", language="python"):        
        self.ds = load_dataset(path, language, split="train")
        print(type(self.ds))
    def prepare_data(self, max_tokens=512):
        # Filter out samples with code longer than max_tokens and those without code or docstring
        self.ds =self.ds.filter(lambda x: len(x['func_code_string'])<max_tokens*4)

        #filter out samples without code or docstring
        self.ds = self.ds.filter(
            lambda x: x['func_code_string'] and x['func_documentation_string']
        )

        # Select only the relevant columns
        self.ds = self.ds.select_columns(["func_code_string", "func_documentation_string"])

        #remove the docstrings which are too short
        self.ds = self.ds.filter(
            lambda x: len(x['func_documentation_string']) > 10
        )

        # Take a subset of the data for training and testing
        

    def format_helper(self, code, docstring):
        
        formatted_string= f"""
        <|user|>\nGenerate a docstring for this python function:\n\n{code}<|end|>\n<|assistant|>\n{docstring}<|end|>
        """
        return {"text": formatted_string}
    
    def format_data(self):
        self.ds = self.ds.map(lambda x: self.format_helper(x['func_code_string'], x['func_documentation_string']))
    
    def save_data(self,train_size=2000, test_size=200, output_dir="phase3_finetuning/data/processed"):
        os.makedirs(output_dir, exist_ok=True)
        shuffled_ds = self.ds.shuffle(seed=42)
        ds_train = shuffled_ds.select(range(train_size))
        ds_test = shuffled_ds.select(range(train_size, train_size + test_size))
        ds_train.save_to_disk(f"{output_dir}/train")
        ds_test.save_to_disk(f"{output_dir}/test")

if __name__ == "__main__":
    handler = dataset_handler()
    handler.prepare_data()
    handler.format_data()
    handler.save_data()

