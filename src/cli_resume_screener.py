import os
import sys
import json
import pandas as pd
import numpy as np
import fitz  
from sentence_transformers import SentenceTransformer, util
import requests
import torch
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import time
from datetime import datetime

# Setup DeepSeek 
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')

# Configuration file path (save in main directory)
CONFIG_FILE = "../config.json"

class ResumeScreenerCLI:
    def __init__(self):
        self.config = self.load_config()
        # Look for resumelist folder in parent directory (main project folder)
        self.resume_folder = "../resumelist"
        
    def load_config(self):
        """Load configuration from file"""
        default_config = {
            "deepseek_api_key": "",
            "resume_folder": "resumelist"
        }
        
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except:
                return default_config
        return default_config
    
    def save_config(self):
        """Save configuration to file"""
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print the main header"""
        print("=" * 60)
        print("Smart Resume Grader")
        print("AI Powered with DeepSeek + PyTorch")
        print("=" * 60)
        print()
    
    def print_menu(self):
        """Print the main menu"""
        print("MAIN MENU")
        print("-" * 30)
        print("1. Start Resume Grading")
        print("2. Instructions")
        print("3. Settings")
        print("4. Credits")
        print("5. Exit")
        print()
    
    def get_user_choice(self, min_choice=1, max_choice=5):
        """Get user input with validation"""
        while True:
            try:
                choice = input("Enter your choice (1-5): ").strip()
                if choice.isdigit() and min_choice <= int(choice) <= max_choice:
                    return int(choice)
                else:
                    print(f"Please enter a number between {min_choice} and {max_choice}")
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                sys.exit(0)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = "\n".join([page.get_text() for page in doc])
            return text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def deepseek_score_resume(self, resume_text, job_description):
        """DeepSeek AI scoring"""
        if not self.config.get("deepseek_api_key"):
            return 0, "DeepSeek API key not configured"
        
        headers = {
            "Authorization": f"Bearer {self.config['deepseek_api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [{
                "role": "user",
                "content": f"""
You are an expert resume evaluator. Rate this resume's fit for the job on a scale of 0-100 and provide a balanced analysis.

Job Description:
{job_description}

Resume:
{resume_text[:3000]}

Instructions:
- Be objective and realistic in your assessment
- Consider both strengths and weaknesses
- Don't overrate candidates - be conservative
- Focus on actual skills and experience match
- Provide specific reasons for your score
- If the candidate is missing a cruical requirement, remove points. No need for unqualified candidates.

Provide only: Score: [number] | Analysis: [balanced explanation highlighting key strengths and areas of concern]
"""
            }],
            "temperature": 0.2
        }
        
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                text = result['choices'][0]['message']['content']
                score_match = re.search(r'Score:\s*(\d+)', text)
                if score_match:
                    score = int(score_match.group(1))
                    return min(score, 100), text
            return 0, f"DeepSeek API error: {response.status_code}"
        except Exception as e:
            return 0, f"DeepSeek API failed: {str(e)}"
    
    def pytorch_skill_analysis(self, resume_text, job_description):
        """Custom PyTorch-based skill analysis with weighted scoring"""
        
        skill_categories = {
            'backend': ['python', 'java', 'node.js', 'express', 'mongodb', 'mysql', 'postgresql'],
            'frontend': ['javascript', 'react', 'angular', 'vue', 'html', 'css'],
            'devops': ['aws', 'azure', 'docker', 'kubernetes', 'jenkins', 'terraform'],
            'ml_ai': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn'],
            'data': ['data analysis', 'sql', 'excel', 'tableau', 'power bi'],
            'tools': ['git', 'github', 'agile', 'scrum'],
            'soft_skills': ['leadership', 'communication', 'problem solving', 'teamwork', 'collaboration']
        }
        
        job_lower = job_description.lower()
        category_weights = {}
        
        for category, skills in skill_categories.items():
            weight = 1.0
            if any(skill in job_lower for skill in skills):
                weight = 2.0
            if category in ['backend', 'frontend'] and any(skill in job_lower for skill in skills):
                weight = 2.5
            category_weights[category] = weight
        
        resume_lower = resume_text.lower()
        found_skills = []
        missing_skills = []
        category_scores = {}
        
        for category, skills in skill_categories.items():
            category_found = []
            category_missing = []
            for skill in skills:
                if skill.lower() in resume_lower:
                    category_found.append(skill)
                    found_skills.append(skill)
                else:
                    category_missing.append(skill)
                    missing_skills.append(skill)
            
            if category_found:
                category_score = (len(category_found) / len(skills)) * category_weights[category] * 100
            else:
                category_score = 0
            category_scores[category] = category_score
        
        # Calculate total weighted score with more conservative scaling
        total_weighted_score = sum(category_scores.values()) / len(category_scores)
        
        # Apply more conservative scaling to skill scores
        total_weighted_score = total_weighted_score * 0.6  # Reduce skill scores by 40%
        
        # Experience detection with more realistic scoring
        experience_score = 0
        experience_details = []
        
        year_patterns = [
            r'(\d+)\+?\s*years?\s*experience',
            r'(\d+)\+?\s*years?\s*in\s*\w+',
            r'experience:\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*of\s*\w+'
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, resume_lower)
            if matches:
                years = max([int(match) for match in matches])
                if years >= 8:
                    experience_score += 18  # Reduced from 25
                    experience_details.append(f"{years}+ years of experience")
                elif years >= 5:
                    experience_score += 12  # Reduced from 20
                    experience_details.append(f"{years} years of experience")
                elif years >= 3:
                    experience_score += 8   # Reduced from 15
                    experience_details.append(f"{years} years of experience")
                elif years >= 1:
                    experience_score += 4   # Reduced from 15
                    experience_details.append(f"{years} year of experience")
                break
        
        project_keywords = ['project', 'developed', 'built', 'created', 'implemented', 'designed']
        if any(keyword in resume_lower for keyword in project_keywords):
            experience_score += 5  # Reduced from 10
            experience_details.append("Real-world projects mentioned")
        
        work_keywords = ['internship', 'intern', 'freelance', 'contract', 'employed', 'worked at']
        if any(keyword in resume_lower for keyword in work_keywords):
            experience_score += 6  # Reduced from 15
            experience_details.append("Work experience detected")
        
        if 'github' in resume_lower or 'portfolio' in resume_lower:
            experience_score += 2  # Reduced from 5
            experience_details.append("GitHub/Portfolio mentioned")
        
        experience_score = min(experience_score, 25)  # Reduced max from 40
        
        # Education detection
        education_score = 0
        education_details = []
        
        education_patterns = [
            (r'bachelor', 10),  # Reduced from 15
            (r'master', 12),    # Reduced from 15
            (r'phd', 15),       # Keep PhD at 15
            (r'degree', 10),    # Reduced from 15
            (r'university', 6), # Reduced from 10
            (r'college', 6),    # Reduced from 10
            (r'diploma', 6),    # Reduced from 10
            (r'certificate', 3), # Reduced from 5
            (r'course', 2),     # Reduced from 5
            (r'bootcamp', 2)    # Reduced from 5
        ]

        for pattern, points in education_patterns:
            if re.search(pattern, resume_lower):
                education_score = max(education_score, points)
                if pattern in ['bachelor', 'master', 'phd']:
                    education_details.append("Formal degree detected")
                elif pattern in ['university', 'college']:
                    education_details.append("Higher education mentioned")
                elif pattern in ['diploma', 'certificate']:
                    education_details.append("Certification/diploma detected")
                break
        
        total_score = (total_weighted_score * 0.5 + experience_score * 0.3 + education_score * 0.2)
        
        summary_parts = []
        suggestions = []
        
        if total_weighted_score > 40:  # Reduced from 60
            summary_parts.append("Strong technical skills demonstrated")
        elif total_weighted_score > 20:  # Reduced from 30
            summary_parts.append("Moderate technical skills shown")
        else:
            summary_parts.append("Limited technical skills mentioned")
            suggestions.append("Add more technical skills relevant to the position")
        
        if experience_score > 20:  # Reduced from 30
            summary_parts.append("Good experience level")
        elif experience_score > 10:  # Reduced from 15
            summary_parts.append("Some experience demonstrated")
        else:
            summary_parts.append("Limited experience mentioned")
            suggestions.append("Include internships, projects, or work experience")
        
        if education_score > 10:
            summary_parts.append("Educational background present")
        else:
            summary_parts.append("No formal education mentioned")
            suggestions.append("Add educational background (even if incomplete)")
        
        if not found_skills:
            suggestions.append("Add technical skills relevant to the position")
        elif len(missing_skills) > len(found_skills):
            suggestions.append("Expand skill set to match job requirements")
        
        if not experience_details:
            suggestions.append("Include internships, freelance work, or personal projects")
        
        if not education_details:
            suggestions.append("Mention educational background or certifications")
        
        summary = f"This resume shows {', '.join(summary_parts).lower()}. "
        if suggestions:
            summary += "Consider: " + "; ".join(suggestions[:3]) + "."
        
        return min(total_score, 100), summary
    
    def combined_ai_scoring(self, resume_text, job_description):
        """Combines DeepSeek API and PyTorch analysis for AI score"""
        
        scores = []
        pytorch_score, _ = self.pytorch_skill_analysis(resume_text, job_description)
        scores.append(pytorch_score)
      
        if self.config.get("deepseek_api_key"):
            deepseek_score, _ = self.deepseek_score_resume(resume_text, job_description)
            if deepseek_score > 0:
                if deepseek_score > 85:
                    deepseek_score = min(deepseek_score, 85)
                
                scores.append(deepseek_score)
        
        if scores:
            final_score = sum(scores) / len(scores)
            
            if final_score > 80:
                final_score = min(final_score, 80)
            
            return final_score, "AI analysis completed"
        
        return 0, "No AI providers available"
    
    def calculate_algorithm_score(self, resume_text, job_description):
        """Calculate algorithm score using multiple methods"""
        
        # Method 1: BERT Score
        job_embedding = model.encode(job_description, convert_to_tensor=True)
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        bert_similarity = util.cos_sim(job_embedding, resume_embedding)[0][0].item()
        
        # Method 2: TF-IDF + Cosine Similarity
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            tfidf_similarity = 0.0
        
        # Method 3: BERT Embeddings
        bert_embeddings_score = bert_similarity * 0.8
        
        # Combine all methods tg
        algorithm_score = (
            bert_similarity * 0.4 +
            tfidf_similarity * 0.3 +
            bert_embeddings_score * 0.3
        ) * 100
        
        return algorithm_score, "Algorithm analysis completed"
    
    def save_ranking_to_txt(self, df_sorted, output_file):
        """Save ranking results to a simple text file"""
        
        # Create result folder if its not there
        result_folder = "../result"
        os.makedirs(result_folder, exist_ok=True)
        
        # Full path for output file
        full_output_path = os.path.join(result_folder, output_file)
        
        # Find the longest name for proper alignment
        max_name_length = max(len(row['Name']) for _, row in df_sorted.iterrows())
        
        with open(full_output_path, 'w', encoding='utf-8') as f:
            f.write("RANK | NAME" + " " * (max_name_length - 4) + " | SCORE\n")
            f.write("-" * (8 + max_name_length + 8) + "\n") 
            
            for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
                name = row['Name']
                final_score = row['FinalScore']
                # if you see this I hope you're having a fantastic day haha :)
              
                # Padding for alignment
                rank_padding = " " * (4 - len(str(i)))
                name_padding = " " * (max_name_length - len(name))
                
                f.write(f"{i}.{rank_padding} | {name}{name_padding} | {final_score:.2f}\n")
        
        return full_output_path
    
    def load_resumes_from_folder(self):
        """Load resumes from the resumelist folder"""
        resumes = []
        resume_path = Path(self.resume_folder)
        
        if not resume_path.exists():
            print(f"Resumelist folder not found. Creating it...")
            os.makedirs(resume_path, exist_ok=True)
            print(f"Created resumelist folder. Please add your resume PDFs to it.")
            return []
        
        pdf_files = list(resume_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in resumelist folder!")
            print(f"Please add resume PDFs to the resumelist folder.")
            return []
        
        print(f"Found {len(pdf_files)} resume(s) in resumelist folder")
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            text = self.extract_text_from_pdf(str(pdf_file))
            if text.strip():
                resumes.append({
                    "Name": pdf_file.stem, 
                    "ResumeText": text
                })
            else:
                print(f"Warning: Could not extract text from {pdf_file.name}")
        
        return resumes
    
    def start_screening(self):
        """Start the resume grading process"""
        self.clear_screen()
        self.print_header()
        
        print("STARTING RESUME GRADING")
        print("=" * 40)
        print()
        
        # Check if DeepSeek API key is configured
        if not self.config.get("deepseek_api_key"):
            print("DeepSeek API Key Setup")
            print("-" * 30)
            print("For enhanced AI analysis, you can configure your DeepSeek API key.")
            print("Get a free API key from: https://platform.deepseek.com/")
            print("IMPORTANT: DeepSeek only shows your API key once at creation!")
            print()
            
            setup_api = input("Would you like to configure DeepSeek API key now? (y/n): ").strip().lower()
            if setup_api in ['y', 'yes']:
                print("\nDeepSeek API Key Configuration")
                print("-" * 40)
                print("Get a free API key from: https://platform.deepseek.com/")
                print("IMPORTANT: DeepSeek only shows your API key once at creation!")
                print("Make sure to copy it immediately and paste it here.")
                print("Leave empty to skip DeepSeek analysis (PyTorch analysis will still work)")
                print()
                
                new_key = input("Enter your DeepSeek API key (or press Enter to skip): ").strip()
                if new_key:
                    self.config["deepseek_api_key"] = new_key
                    self.save_config()
                    print("API key saved successfully!")
                    print("Your API key is now stored and will be used automatically.")
                else:
                    print("Skipping DeepSeek API configuration.")
                    print("PyTorch analysis will still work for resume grading.")
                
                input("\nPress Enter to continue...")
                self.clear_screen()
                self.print_header()
                print("STARTING RESUME GRADING")
                print("=" * 40)
                print()
        
        # Load resumes
        resumes = self.load_resumes_from_folder()
        if not resumes:
            input("\nPress Enter to return to main menu...")
            return
        
        print("JOB DESCRIPTION")
        print("-" * 20)
        print("Please paste the job description below (press Enter twice when done):")
        print()
        
        job_lines = []
        while True:
            line = input()
            if line == "" and job_lines and job_lines[-1] == "":
                break
            job_lines.append(line)
        
        job_description = "\n".join(job_lines[:-1])  # Remove the last empty line
        
        if not job_description.strip():
            print("No job description provided!")
            input("\nPress Enter to return to main menu...")
            return
        
        print(f"\nJob description loaded ({len(job_description)} characters)")
        print()
        
        print("PROCESSING RESUMES...")
        print("This may take a few minutes...")
        print()
        
        df = pd.DataFrame(resumes)
        df['AlgorithmScore'] = 0.0
        df['AIScore'] = 0.0
        
        for i in range(len(df)):
            print(f"Analyzing {df.loc[i, 'Name']}...")
            
            # Algorithm Score
            algorithm_score, _ = self.calculate_algorithm_score(
                df.loc[i, 'ResumeText'], job_description
            )
            df.at[i, 'AlgorithmScore'] = algorithm_score
            
            # AI Score
            ai_score, _ = self.combined_ai_scoring(
                df.loc[i, 'ResumeText'], job_description
            )
            df.at[i, 'AIScore'] = ai_score
        
        # Calculate final score with improved scaling
        df['FinalScore'] = 0.6 * df['AIScore'] + 0.4 * df['AlgorithmScore']
        
        # Apply more realistic scaling to prevent overrating
        df['FinalScore'] = df['FinalScore'] * 0.7  # Reduce overall scores by 30%
        
        # Ensure scores don't exceed realistic bounds
        df['FinalScore'] = df['FinalScore'].clip(upper=85)  # Cap at 85
        
        df_sorted = df.sort_values(by='FinalScore', ascending=False)
        
        # Save ranking as a .txt
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_output = f"resume_ranking_{timestamp}.txt"
        
        print("Saving ranking results...")
        output_path = self.save_ranking_to_txt(df_sorted, txt_output)
        self.clear_screen()
        self.print_header()
        
        print("RESUME GRADING COMPLETE!")
        print("=" * 40)
        print()
        print(f"Resume ranking has been saved to: {txt_output}")
        print(f"Location: result/{txt_output}")
        print()
        print("Open the text file to view the ranking results!")
        print()
        
        print("Closing in 3 seconds...")
        time.sleep(3)
    
    def show_instructions(self):
        """Show instructions"""
        self.clear_screen()
        self.print_header()
        
        print("INSTRUCTIONS")
        print("=" * 30)
        print()
        print("How to use Smart Resume Grader:")
        print()
        print("1. Prepare Resumes:")
        print("   - Add your resume PDF files to the resumelist folder")
        print("   - Supported format: PDF only")
        print()
        print("2. Configure Settings (Optional):")
        print("   - Go to Settings in the main menu")
        print("   - Add your DeepSeek API key for enhanced AI analysis")
        print("   - Get free API key from: https://platform.deepseek.com/")
        print("   - API key is automatically saved and reused!")
        print()
        print("3. Start Grading:")
        print("   - Select 'Start Resume Grading' from the main menu")
        print("   - First time? You'll be prompted to configure DeepSeek API key")
        print("   - Paste the job description when prompted")
        print("   - Wait for the AI analysis to complete")
        print("   - Review the ranked results")
        print()
        print("4. Understanding Scores:")
        print("   - Final Score: Combined AI and Algorithm analysis (60% AI + 40% Algorithm)")
        print("   - AI Score: DeepSeek AI + PyTorch ML analysis")
        print("   - Algorithm Score: BERT + TF-IDF + Cosine similarity")
        print()
        print("5. Results:")
        print("   - Results are automatically saved as text files in the result folder")
        print("   - Simple ranking format with final scores")
        print()
        print("Technical Features:")
        print("   - AI-powered analysis using DeepSeek API")
        print("   - Custom PyTorch ML models for skill analysis")
        print("   - BERT embeddings for semantic similarity")
        print("   - TF-IDF and cosine similarity algorithms")
        print("   - Weighted scoring system with safeguards against overrating")
        print()
        
        input("Press Enter to return to main menu...")
    
    def show_settings(self):
        """Show and edit settings"""
        while True:
            self.clear_screen()
            self.print_header()
            
            print("⚙️  SETTINGS")
            print("=" * 20)
            print()

            # Display settings
            api_key_status = "Configured" if self.config.get("deepseek_api_key") else "Not configured"
            
            print(f"1. DeepSeek API Key: {api_key_status}")
            print("2. Back to Main Menu")
            print()
            
            choice = self.get_user_choice(1, 2)
            
            if choice == 1:
                print("\nDeepSeek API Key Configuration")
                print("-" * 40)
                print("Get a free API key from: https://platform.deepseek.com/")
                print("IMPORTANT: DeepSeek only shows your API key once at creation!")
                print("Make sure to copy it immediately and paste it here.")
                print("Leave empty to skip DeepSeek analysis (PyTorch analysis will still work)")
                print()
                
                new_key = input("Enter your DeepSeek API key (or press Enter to skip): ").strip()
                if new_key:
                    self.config["deepseek_api_key"] = new_key
                    self.save_config()
                    print("API key saved successfully!")
                    print("Your API key is now stored and will be used automatically.")
                else:
                    self.config["deepseek_api_key"] = ""
                    self.save_config()
                    print("API key cleared!")
                
                input("\nPress Enter to continue...")
                
            elif choice == 2:
                break
    
    def show_credits(self):
        """Show credits"""
        self.clear_screen()
        self.print_header()
        
        print("CREDITS")
        print("=" * 20)
        print()
        print("Smart Resume Grader")
        print("AI Powered with DeepSeek + PyTorch")
        print()
        print("Made by: Praagya Lamsal")
        print()
        print("Connect with me:")
        print("   LinkedIn: https://www.linkedin.com/in/praagya-lamsal/")
        print("   GitHub: https://github.com/PraagyaLamsal/")
        print()
        print("Technologies Used:")
        print("   - Python")
        print("   - DeepSeek AI API")
        print("   - PyTorch")
        print("   - scikit-learn")
        print("   - BERT embeddings")
        print()
        print("License: MIT")
        print("Open Source Project")
        print()
        
        input("Press Enter to return to main menu...")
    
    def run(self):
        """Main application loop"""
        while True:
            self.clear_screen()
            self.print_header()
            self.print_menu()
            
            choice = self.get_user_choice()
            
            if choice == 1:
                self.start_screening()
            elif choice == 2:
                self.show_instructions()
            elif choice == 3:
                self.show_settings()
            elif choice == 4:
                self.show_credits()
            elif choice == 5:
                print("\nThank you for using Smart Resume Grader!")
                print("Goodbye!")
                break

def main():
    """Main entry point"""
    try:
        screener = ResumeScreenerCLI()
        screener.run()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main() 
