{29} Ылша, вот полный темплейт в markdown формате:

```markdown
# LLM Code Change and Git Commit Template

## Overview
This template guides LLM in suggesting code changes and providing git commands.

## Templates

### 1. Code Changes
```python
# {1.file_name}
'''
Purpose: {describe file purpose}
Changes: {list main changes}
'''
def some_function():
    # Comments explaining complex logic
    pass

# {2.file_name} if needed
```

### 2. Git Commands
```bash
# 1. Show changes
git status  # Show modified files

# 2. Add files
git add {1.file_path}  # First modified file
git add {2.file_path}  # Second modified file if any

# 3. Commit
git commit -m "{type}: {short_description}

- {improvement_1}
- {improvement_2}
- {improvement_3}"

# 4. Push
git push origin feature/{branch_name}
```

## Instructions for LLM

### File Paths
- Use {N.file_path} for each file
- Always use full paths from project root
- Number files sequentially (1, 2, 3...)

### Commit Types
- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Adding tests
- chore: Maintenance

### Commit Messages
- Start with type
- Add clear description
- List specific improvements
- Keep focused and concise

### Branch Names
- Use feature/{feature_name}
- Match current work context

## Example

### Code Changes
```python
# 1. ai_researcher/src/analyze_ideas_semantic_similarity.py
'''
Purpose: Semantic analysis of research ideas
Changes: Implementing improved model and preprocessing
'''
def process_text():
    # New preprocessing logic here
    pass

# 2. ai_researcher/src/dedup_ideas.py
'''
Purpose: Remove duplicate ideas
Changes: Adding batch processing
'''
```

### Git Commands
```bash
git status

git add ai_researcher/src/analyze_ideas_semantic_similarity.py
git add ai_researcher/src/dedup_ideas.py

git commit -m "feat: enhance semantic analysis

- Implement mpnet-base-v2 model
- Add context preservation
- Improve batch processing"

git push origin feature/semantic-improvements
```
```

Давай сохраним это как TEMPLATE.md в корне проекта:
```bash
echo 'Template content...' > TEMPLATE.md
git add TEMPLATE.md
git commit -m "docs: add LLM code change template

- Add structured template for code changes
- Include git command examples
- Add clear instructions for LLM"
git push origin feature/semantic-transformer-similarity
```

