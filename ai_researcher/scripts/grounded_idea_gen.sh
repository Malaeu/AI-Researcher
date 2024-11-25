## example usage
topic_names=("novelty_automation_framework")
ideas_n=5 ## batch size
methods=("prompting")
rag_values=("true" "false")

# Iterate over each seed (increased to generate ~500 ideas total)
for seed in {1..50}; do
    # Iterate over each topic name 
    for topic in "${topic_names[@]}"; do
        # Iterate over each method
        for method in "${methods[@]}"; do
            # Iterate over RAG values true and false
            for rag in "${rag_values[@]}"; do
                echo "Running grounded_idea_gen.py on: $topic with seed $seed and RAG=$rag"
                python3 src/grounded_idea_gen.py \
                 --engine "claude-3-5-sonnet-20241022" \
                 --paper_cache "../cache_results_test/lit_review/$topic.json" \
                 --idea_cache "../cache_results_test/seed_ideas/$topic.json" \
                 --grounding_k 10 \
                 --method "$method" \
                 --ideas_n $ideas_n \
                 --RAG "$rag" \
                 --seed "$seed"
            done
        done
    done
done

# After generating all ideas, run deduplication
echo "Running deduplication on generated ideas..."
bash scripts/dedup_ideas.sh
