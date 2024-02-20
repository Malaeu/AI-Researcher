from openai import OpenAI
from utils import call_api
import argparse
import json
import os
from lit_review_tools import format_papers_for_printing
from utils import cache_output
import retry
from tqdm import tqdm
import random 
random.seed(2024)

@retry.retry(tries=3, delay=2)
def plan_generation(idea, demo_examples, topic_description, openai_client, model, seed):
    ## forumate an idea from a paragraph into a full experiment plan based on our template

    prompt = "You are an expert researcher in Natural Language Processing and your job is to expand a vague project idea into a detailed experiment plan. I will provide you with an idea on the topic of: " + topic_description + ".\n\n"
    prompt += "The idea is:\n" + idea + "\n\n"
    # prompt += "Here are some relevant papers that are relevant to the idea:\n" + format_papers_for_printing(grounding_papers) + "\n\n"
    prompt += "Now you should come up with the full proposal covering:\n"
    prompt += "1. Title: A concise statement of the main research question\n"
    prompt += "2. Problem Statement: Clearly define the problem your research intends to address. Explain clearly why this problem is interesting and meaningful.\n"
    prompt += "3. Step-by-Step Experiment Plan: Break down every single step of the experiments, make sure every step is executable. Cover all essential details such as the datasets, models, and metrics to be used. If the project involves proposing a new method, make sure to expand it in detail.\n"
    prompt += "You should specify the intended type of contribution in the problem statement, which should be either: 1) analysis; or 2) method. Analysis papers should state clearly the type of model behavior that we are studying, and why such behaviour would be interesting to study. You can also list hypotheses on what possible patterns we might expect and what are the corresponding implications. Ideally, for analysis papers, we want the results to be interesting no matter which hypothesis turns out to be true.\n"
    prompt += "Method papers should state clearly the new method being proposed, the motivation behind the new method, and the concrete success criteria (i.e., if the proposed method manages to achieve this, then the method is shown to be useful). For method papers, you should also include a fallback option: what should the students do if the method didn't manage to satisfy the success criteria. For example, you can suggest additional analysis to help debug why the proposed method didn't work, which could inform alternative new methods, or just turn the project into an analysis paper instead by offering some interesting insights.\n"
    prompt += "The experiment plan should just focus on the experiments, and should not include any introduction or background information (e.g., you should skip the literature review, paper writing plan, and ethical discussion). Just give instructions on the experiments.\n"
    prompt += "When designing experiments, note that the goal is to have a short-term project that can be finished within two weeks. So, try to avoid training models from scrach and instead prefer prompting-based methods. On rare cases, finetuning small open-source language models is also acceptable. Try to avoid any human evaluation or human data collection, which is time-consuming and expensive. The focus can be either to improve model performance on same tasks or to perform interesting analysis and reveal insights.\n"
    prompt += "Below is a few examples for your reference:\n"
    prompt += demo_examples + "\n\n"
    prompt += "Now please write down your experiment plan (each section should be described as a few concise sentences; index the sections and separate them with new lines). Make sure to be as detailed as possible especially for the data and methods, so that a student can directly follow the plan to implement the experiment. For example, if the method involves prompting, describe the prompts in detail. Give prompt or algorithm description for all methods involved."
    
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=4096, seed=seed, json_output=False)
    return prompt, response, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default=None, required=True, help='cache file name for the retrieved papers')
    parser.add_argument('--idea_name', type=str, default=None, required=True, help='the specific idea to be formulated into an experiment plan')
    parser.add_argument('--grounding_k', type=int, default=10, help='how many papers to use for grounding')
    parser.add_argument('--seed', type=int, default=2024, help="seed for GPT-4 generation")
    args = parser.parse_args()

    with open("keys.json", "r") as f:
        keys = json.load(f)

    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    S2_KEY = keys["s2_key"]
    openai_client = OpenAI(
        organization=ORG_ID,
        api_key=OAI_KEY
    )

    ## load the grounding papers
    with open(os.path.join("cache_results/lit_review", args.cache_name+".json"), "r") as f:
        lit_review = json.load(f)
    topic_description = lit_review["topic_description"]
    paper_bank = lit_review["paper_bank"]
    grounding_papers = paper_bank[ : args.grounding_k]

    ## load the demo examples
    with open("experiment_plan_examples.txt", "r") as f:
        demo_examples = f.read().strip()

    if args.idea_name == "all":
        ## load all ideas 
        with open(os.path.join("cache_results/ideas", args.cache_name+".json"), "r") as f:
            ideas = json.load(f)["ideas"]
        
        for idea_name, idea in tqdm(ideas.items()):
            prompt, response, cost = plan_generation(idea, demo_examples, topic_description, openai_client, args.engine, args.seed)
            print (idea_name)
            print (response)
            print ("Total cost: ", cost)
            print ("\n")
        
            ## save the cache
            if not os.path.exists("cache_results/experiment_plans/"+args.cache_name):
                os.makedirs("cache_results/experiment_plans/"+args.cache_name)
            cache_dict = {"topic_description": topic_description, "idea_name": idea_name, "raw_idea": idea, "experiment_plan": response.strip()}
            cache_file = os.path.join("cache_results/experiment_plans/"+args.cache_name, "_".join(idea_name.lower().split())+".json")
            cache_output(cache_dict, cache_file)

    else:
        ## load the idea 
        with open(os.path.join("cache_results/ideas", args.cache_name+".json"), "r") as f:
            ideas = json.load(f)["ideas"]
        idea = ideas[args.idea_name]

        prompt, response, cost = plan_generation(idea, demo_examples, topic_description, openai_client, args.engine, args.seed)
        print (response)
        print ("Total cost: ", cost)

        ## save the cache
        if not os.path.exists("cache_results/experiment_plans/"+args.cache_name):
            os.makedirs("cache_results/experiment_plans/"+args.cache_name)
        cache_dict = {"topic_description": topic_description, "idea_name": args.idea_name, "raw_idea": idea, "experiment_plan": response.strip()}
        cache_file = os.path.join("cache_results/experiment_plans/"+args.cache_name, "_".join(args.idea_name.lower().split())+".json")
        cache_output(cache_dict, cache_file)

    