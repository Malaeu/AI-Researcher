from openai import OpenAI
import anthropic
from together import Together
from utils import call_api, shuffle_dict_and_convert_to_string
import argparse
import json
import os
from lit_review_tools import format_papers_for_printing
from utils import cache_output
import random 
import retry
import sys

def validate_api_keys(engine):
    """Validate API keys based on the selected engine."""
    try:
        keys_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "keys.json")
        if not os.path.exists(keys_file):
            raise FileNotFoundError(f"API keys file not found at {keys_file}")
            
        with open(keys_file, "r") as f:
            keys = json.load(f)
            
        # Only check keys needed for the selected engine
        required_keys = {}
        if "claude" in engine.lower():
            required_keys["anthropic_key"] = "Anthropic"
        elif "gpt" in engine.lower():
            required_keys["api_key"] = "OpenAI"
        elif "llama" in engine.lower():
            required_keys["together_key"] = "Together"
        
        print(f"\nValidating API keys for engine: {engine}")
        
        missing_keys = []
        for key, provider in required_keys.items():
            if key not in keys or not keys[key]:
                missing_keys.append(provider)
                
        if missing_keys:
            print(f"Error: Missing required API key for {engine}: {', '.join(missing_keys)}")
            print(f"Please add your {', '.join(missing_keys)} API key to ai_researcher/keys.json")
            sys.exit(1)
        else:
            print("API key validation successful")
            
        return keys
            
    except json.JSONDecodeError as e:
        print(f"Error parsing keys.json: {str(e)}")
        print("Please ensure the file is valid JSON format")
        raise
    except Exception as e:
        print(f"Error validating API keys: {str(e)}")
        raise

def load_cache_file(cache_path):
    try:
        if not os.path.exists(cache_path):
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            # Create empty cache file
            with open(cache_path, "w") as f:
                json.dump({}, f)
            print(f"Created new cache file at {cache_path}")
            return {}
            
        with open(cache_path, "r") as f:
            return json.load(f)
            
    except json.JSONDecodeError as e:
        print(f"Error parsing cache file {cache_path}: {str(e)}")
        print("Creating new empty cache file")
        with open(cache_path, "w") as f:
            json.dump({}, f)
        return {}
    except Exception as e:
        print(f"Error loading cache file {cache_path}: {str(e)}")
        raise

@retry.retry(tries=3, delay=2)
def idea_generation(method, existing_ideas, paper_bank, grounding_k, examples, ideas_n, topic_description, openai_client, model, seed, temperature, max_tokens, RAG=True):
    try:
        top_papers = paper_bank[:int(grounding_k * 2)]
        random.shuffle(top_papers)
        grounding_papers = top_papers[:grounding_k]

        prompt = "You are a creative researcher. Your task is to propose " + str(ideas_n) + " novel research ideas related to: " + topic_description + ".\n\n"
        
        if existing_ideas:
            prompt += "Here are some existing ideas that have already been proposed (avoid repeating these):\n" + existing_ideas + "\n\n"
        
        if RAG:
            prompt += "Here are some relevant papers on this topic just for your background knowledge:\n" + format_papers_for_printing(grounding_papers, include_score=False, include_id=False) + "\n"
        
        prompt += "\nHere are some examples of how to format your response:\n" + examples + "\n"
        prompt += "\nNow propose " + str(ideas_n) + " novel research ideas. Make sure each idea has a clear title, description, and methodology. Make sure to follow the exact same JSON format as shown in the examples.\n"

        prompt_messages = [{"role": "user", "content": prompt}]
        
        try:
            response, cost = call_api(openai_client, model, prompt_messages, temperature=temperature, max_tokens=max_tokens, seed=seed, json_output=True)
            return prompt, response, cost
        except Exception as e:
            print(f"API call failed: {str(e)}")
            print("Prompt that failed:")
            print(prompt)
            raise
            
    except Exception as e:
        print(f"Error in idea generation preparation: {str(e)}")
        raise

def generate_ideas(lit_review, idea_cache, client, engine, method, RAG, temperature, ideas_n, grounding_k):
    try:
        topic_description = lit_review["topic_description"]
        paper_bank = lit_review["paper_bank"]

        if RAG:
            print("RAG is enabled for idea generation")
        else:
            print("RAG is disabled for idea generation")

        print("topic: ", topic_description)

        # Extract existing ideas
        existing_ideas = None
        if "ideas" in idea_cache:
            existing_ideas = [key for idea in idea_cache["ideas"] for key in idea.keys()]
            existing_ideas = list(set(existing_ideas))
            existing_ideas = "; ".join(existing_ideas)
        
        print("existing ideas: ", existing_ideas)
        print("\n")
        print(f"generating {ideas_n} ideas...")

        # Load method-specific examples
        try:
            if method == "prompting":
                with open("prompts/idea_examples_prompting_method.json", "r") as f:
                    method_idea_examples = json.load(f)
                    method_idea_examples = shuffle_dict_and_convert_to_string(method_idea_examples)
            elif method == "finetuning":
                with open("prompts/idea_examples_finetuning_method.json", "r") as f:
                    method_idea_examples = json.load(f)
                    method_idea_examples = shuffle_dict_and_convert_to_string(method_idea_examples)
            else:
                with open("prompts/idea_examples_method.json", "r") as f:
                    method_idea_examples = json.load(f)
                    method_idea_examples = shuffle_dict_and_convert_to_string(method_idea_examples, n=4)
        except FileNotFoundError as e:
            print(f"Error loading example file: {str(e)}")
            print("Current working directory:", os.getcwd())
            print("Available files in prompts directory:")
            try:
                print(os.listdir("prompts"))
            except:
                print("Could not list prompts directory")
            raise

        # Generate ideas
        prompt, response, cost = idea_generation(
            method=method,
            existing_ideas=existing_ideas,
            paper_bank=paper_bank,
            grounding_k=grounding_k,
            examples=method_idea_examples,
            ideas_n=ideas_n,
            topic_description=topic_description,
            openai_client=client,
            model=engine,
            seed=2024,
            temperature=temperature,
            max_tokens=30000,
            RAG=RAG
        )

        print("idea generation cost: ", cost)
        
        # Parse and save response
        response = json.loads(response.strip())
        ideas = {"topic_description": topic_description, "ideas": [response]}
        
        if "ideas" in idea_cache:
            idea_cache["ideas"].append(response)
            ideas = idea_cache
        
        print("#ideas generated so far: ", sum(len(d) for d in ideas["ideas"]))

        # Save to cache
        cache_dir = os.path.dirname(args.idea_cache)
        os.makedirs(cache_dir, exist_ok=True)
        cache_output(ideas, args.idea_cache)

    except Exception as e:
        print("Error in idea generation:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='claude-3-5-sonnet-20241022', help='api engine')
    parser.add_argument('--paper_cache', type=str, help='cache file for papers')
    parser.add_argument('--idea_cache', type=str, help='cache file for ideas')
    parser.add_argument('--grounding_k', type=int, default=10, help='how many papers for grounding')
    parser.add_argument('--method', type=str, default='prompting', help='method for generating ideas')
    parser.add_argument('--RAG', type=str, default='true', help='whether to use RAG (true/false)')
    parser.add_argument('--temperature', type=float, default=0.9, help='temperature in sampling')
    parser.add_argument('--ideas_n', type=int, default=5, help="how many ideas to generate")
    parser.add_argument('--seed', type=int, default=2024, help="seed for generation")
    args = parser.parse_args()

    try:
        print("\n=== Starting Idea Generation ===")
        print(f"Engine: {args.engine}")
        print(f"Method: {args.method}")
        print(f"RAG enabled: {args.RAG}")
        print(f"Generating {args.ideas_n} ideas with seed {args.seed}")
        
        # Load and validate API keys
        keys = validate_api_keys(args.engine)
        
        # Load paper cache file (must exist)
        print("\nLoading paper cache...")
        lit_review = load_cache_file(args.paper_cache)
        if lit_review is None:
            print("Error: Failed to load paper cache file")
            sys.exit(1)
        print(f"Successfully loaded {len(lit_review)} papers from cache")
        
        # Load or create idea cache file
        print("\nLoading/creating idea cache...")
        idea_cache = load_cache_file(args.idea_cache)
        if idea_cache is None:
            print("Error: Failed to create idea cache file")
            sys.exit(1)
        print("Successfully loaded idea cache")
        
        # Set random seed
        random.seed(args.seed)
        
        # Convert RAG string to boolean
        use_rag = args.RAG.lower() == 'true'
        
        # Initialize clients based on engine
        print("\nInitializing AI model client...")
        if "claude" in args.engine.lower():
            if "anthropic_key" not in keys or not keys["anthropic_key"].strip():
                print(f"Error: Anthropic API key is required for engine '{args.engine}'")
                print("Please add your Anthropic API key to ai_researcher/keys.json")
                sys.exit(1)
            client = anthropic.Anthropic(api_key=keys["anthropic_key"])
            print("Successfully initialized Anthropic client")
        else:
            if "api_key" not in keys or not keys["api_key"].strip():
                print(f"Error: OpenAI API key is required for engine '{args.engine}'")
                print("Please add your OpenAI API key to ai_researcher/keys.json")
                sys.exit(1)
            client = OpenAI(api_key=keys["api_key"])
            print("Successfully initialized OpenAI client")
        
        # Generate ideas with proper error handling
        try:
            generate_ideas(
                lit_review=lit_review,
                idea_cache=idea_cache,
                client=client,
                engine=args.engine,
                method=args.method,
                RAG=use_rag,
                temperature=args.temperature,
                ideas_n=args.ideas_n,
                grounding_k=args.grounding_k
            )
        except Exception as e:
            print(f"Error in idea generation: {str(e)}")
            import traceback
            print("Traceback:")
            print(traceback.format_exc())
            sys.exit(1)
            
    except FileNotFoundError:
        print("keys.json file not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in keys.json: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        sys.exit(1)
