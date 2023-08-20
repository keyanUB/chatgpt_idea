import jsonlines
import pandas as pd
import openai
import time

from tqdm import tqdm

def main(
    data_path: str = "data/implicit_hate.csv",
    output_path: str = "data/implicit_hate_v1_gpt3.jsonl",     
):
    # load the dataset
    print("Loading dataset...")

    # # load the data
    # with jsonlines.open(data_path) as reader:
    #     data = [line for line in reader]
    #     print("data loaded from {}".format(data_path))
    pd_data = pd.read_csv(data_path)
    print("data loaded from {}".format(data_path))
    print("data shape: {}".format(pd_data.shape))

    # load the model
    openai.api_key = "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"

    def get_prompt(text):
        # prompt = f"Consider the following comment, and carefully answer the questions in each step to conclude whether it is hate speech or not: \ncomment:\"{text}\"\nLet's think of it in the following steps:\nQ1A: Which of the following identities are mentioned in the text?\n'race', 'nationality', 'age', 'political', 'religion', ‘disability’, '(anti-)masker', '(anti-)vaxxer'\nQ1B: Are there any individuals mentioned explicitly by their name, answer 'Yes' or 'No'?\nQ2: Are there any derogatory, humiliating, insulting, or disparaging words or phrases specifically mentioned in the text? Answer 'Yes' or 'No'. (Note: Colloquially usage of the words should not be considered)\nQ3A: If Q2's answer is 'Yes', are those words or phrases directed towards or targeting your selected identities? Answer 'Yes' or 'No'.\nQ3B: If Q2's answer is 'Yes', are those words or phrases directed towards or targeting individuals?  Answer 'Yes' or 'No'.\nQ4A: If Q3A's answer is 'Yes', do those terms incite hate against the selected identities?  Answer 'Yes' or 'No'.\nQ4B: If Q3B's answer is 'Yes', do those terms incite hate against the individual?  Answer 'Yes' or 'No'.\nQ5A: If Q4A’s 'Yes', the comment can be concluded as identity hate speech. Tell me your final conclusion: 'Yes' or 'No'.\nQ5B: If Q4B’s 'Yes', the comment can be concluded as individual hate speech. Tell me your final conclusion: 'Yes' or 'No'.\nShow your answers in the following format: [‘Q1A’: <answer>, ‘Q1B’: <answer>, ‘Q2’: <answer>, ‘Q3A’: <answer>, ‘Q3B’: <answer>, ‘Q4A’: <answer>, ‘Q4B’: <answer>, ‘Q5A’: <answer>, ‘Q5B’: <answer>]"
        prompt = """
Do you think this comment is hateful? 
a. Yes b. No
""".format(text=text)  
        return prompt
    
    # test with gpt3.5 model
    def prompt_completion(prompt):
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": prompt}]
            )
            sentence  = completion.choices[0]['message']['content']
            time.sleep(0.5)
        
        except Exception as e:
            sentence = f"error: {e}"
            time.sleep(20)

        return sentence
    
    # generate the results
    i = 0
    # for line in data:
    outputs = []
    for id, row in tqdm(pd_data.iterrows()):
        post = row['post']

        prompt = get_prompt(post)
        result = prompt_completion(prompt)
        # print("prompt: {}".format(prompt))
        # print("result: {}".format(result))
        output = {'index': id,  'text': post,  'result': result}
        outputs.append(output)

        i += 1
        # save the results
        if i %100 ==0 or i == len(pd_data):
            with jsonlines.open(output_path, mode='w') as writer:
                writer.write(outputs)
                print("results saved to {}".format(output_path))

if __name__ == "__main__":
    main()