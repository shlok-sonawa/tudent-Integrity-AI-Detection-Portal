Student Integrity Assessment Portal

A Streamlit-based application that compares two student submissions and calculates their code similarity. It also provides a basic AI-footprint score based on predefined code patterns.

Features

Upload Python (.py) or text (.txt) files

Paste code or text directly

Compare two student submissions

Normalize Python variable, function, and parameter names using AST

Calculate similarity using TF-IDF and Cosine Similarity

Generate a basic AI-footprint score

Display results through a simple Streamlit interface

Technologies Used

Python

Streamlit

Scikit-learn

Python AST

TF-IDF

Cosine Similarity

How It Works

Two student submissions are provided through file upload or text input.

Python code is parsed using the AST module.

Variable, function, and parameter names are normalized.

TF-IDF converts the submissions into numerical vectors.

Cosine Similarity calculates the similarity percentage.

A basic heuristic analysis generates an AI-footprint score.

Installation

Clone the repository:

git clone https://github.com/your-username/student-integrity-assessment.git
cd student-integrity-assessment


Install the required packages:

pip install -r requirements.txt


Run the application:

streamlit run app.py

Requirements

Create a requirements.txt file with:

streamlit
scikit-learn

Project Structure
student-integrity-assessment/
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE

Important Note

The similarity score is an automated screening result and does not by itself prove plagiarism.

The AI-footprint score is based on simple heuristics and should not be considered a reliable AI detector. Results should be reviewed by a teacher or instructor before making any academic decision.

Future Improvements

Support for more programming languages

Compare multiple submissions

Generate reports

Add a database

Improve code similarity analysis

Improve AI-generated code detection

Author

Your Name
