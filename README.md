# Smart Resume Grader

AI Powered with DeepSeek + PyTorch

## Features

- **AI-Powered Analysis**: Uses DeepSeek AI API for intelligent resume evaluation
- **PyTorch ML Models**: ML Models for skill analysis/education match
- **BERT Embeddings**: Semantic similarity analysis using BERT
- **TF-IDF + Cosine Similarity**: Traditional NLP algorithms for comparison
- **Weighted Scoring**: Combines multiple analysis methods with safeguards, assigned weight to skills.
- **PDF Processing**: Automatic text extraction from PDF resumes.
- **Batch Processing**: Process multiple resumes at once.

## Quick Start

1. **Install dependencies**:
   Double-click `install_requirements.bat`. (Reccomended)
   
   OR

   ```bash
   pip install -r src/requirements.txt
   ```

1. **Add resumes**:
   - Create a new folder in the main directory called `resumelist`.
   - Place your resume files in the `resumelist` folder.
   - Make sure the files are .pdf files. The reccomended maximum is 50 files at once

3. **Run the program**:
   ```bash
   python src/cli_resume_screener.py
   ```
   Or double-click `run_screener.bat` (Reccomended)

4. **Follow the menu**:
   - Select "Start Resume Grading"
   - Paste the job description when prompted
   - Wait for analysis to complete
   - Results saved to `result/` folder

## Configuration

### DeepSeek API Key (Optional)
- Get free API key from: https://platform.deepseek.com/
- Enhances analysis with AI evaluation
- Works without API key (PyTorch analysis only; Not reccomended)

## Scoring System

### Final Score (60% AI + 40% Algorithm)
- **AI Score**: DeepSeek AI + PyTorch ML analysis
- **Algorithm Score**: BERT + TF-IDF + Cosine similarity

### Analysis Components
- **Skill Analysis**: Technical skills matching job requirements
- **Experience Analysis**: Years of experience, projects, work history
- **Education Analysis**: Formal education and certifications
- **Semantic Similarity**: Content relevance to job description

## Output

- **Text File**: Simple ranking table with aligned columns
- **Location**: `result/resume_ranking_YYYYMMDD_HHMMSS.txt`
- **Format**: RANK | NAME | SCORE

## Contact Me

- **LinkedIn**: linkedin.com/in/praagya-lamsal/
