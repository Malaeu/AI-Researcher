## example usage
cache_dir="../cache_results_test/seed_ideas/"
output_dir="../cache_results_test/ideas_dedup/"
cache_names=("novelty_automation_framework")

for cache_name in "${cache_names[@]}"; do
    echo "Running analyze_ideas_semantic_similarity.py with cache_name: $cache_name"
    python3 src/analyze_ideas_semantic_similarity.py \
        --cache_dir "$cache_dir" \
        --cache_name "$cache_name" \
        --save_similarity_matrix
done

for cache_name in "${cache_names[@]}"; do
    echo "Running dedup_ideas.py with cache_name: $cache_name"
    python3 src/dedup_ideas.py \
        --cache_dir "$cache_dir" \
        --cache_name "$cache_name" \
        --dedup_cache_dir "$output_dir" \
        --similarity_threshold 0.8 
done
