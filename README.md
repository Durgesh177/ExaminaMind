ExaminaMind: A Cognitive AI Framework for Predictive Examination Analytics

ExaminaMind is an AI-powered examination analytics system designed to analyze historical examination question papers and identify recurring academic topics.

The system extracts text from uploaded question papers, preprocesses the content using Natural Language Processing techniques, and applies TF-IDF vectorization and Logistic Regression to perform topic classification and predictive analysis.

Features
Upload examination question papers
Supports PDF, DOCX, and TXT files
OCR fallback for scanned PDF documents
Automatic text extraction
NLP-based text preprocessing
Keyword-based weak supervision for topic labeling
TF-IDF feature extraction
Logistic Regression-based topic classification
Topic prediction with confidence scores
Examination analytics and visualization pages
Download and delete uploaded files


Machine Learning Workflow

Question Papers
       ↓
Text Extraction
       ↓
Text Preprocessing
       ↓
Keyword-Based Weak Supervision
       ↓
Topic Label Generation
       ↓
TF-IDF Vectorization
       ↓
Logistic Regression
       ↓
Topic Prediction
       ↓
Confidence Scores & Analytics

Supported Topics

The current version supports classification across the following topics:

Artificial Intelligence
Machine Learning
DBMS
Operating Systems
Computer Networks
Data Structures
Software Engineering
Computer Architecture
Cyber Security
Technologies Used
Backend
Python
Flask
Machine Learning & NLP
Scikit-learn
TF-IDF Vectorization
Logistic Regression
NLTK
Document Processing
PyMuPDF
python-docx
pytesseract
Pillow

🛠️ Technologies Used
Backend
Python
Flask
Machine Learning & NLP
Scikit-learn
TF-IDF Vectorization
Logistic Regression
NLTK
Document Processing
PyMuPDF
python-docx
pytesseract
Pillow

📄 Supported File Formats

ExaminaMind currently supports:

.pdf
.docx
.txt

For scanned PDF documents, the application uses OCR as a fallback.

🔍 How It Works
Upload historical examination question papers.
ExaminaMind extracts text from the uploaded documents.
The extracted text is cleaned and preprocessed.
Keyword-based weak supervision generates topic labels.
TF-IDF converts the text into numerical features.
Logistic Regression learns topic classification patterns.
The system generates topic predictions and confidence scores.
Results are displayed through prediction and visualization pages.


ExaminaMind/
│
├── app.py
├── model.py
├── utils.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── .gitkeep
│
├── uploads/
│   └── .gitkeep
│
├── static/
│
└── templates/


🔮 Future Improvements
Split examination papers into individual questions
Train using manually labeled question-topic datasets
Analyze topic frequency across multiple examination years
Detect recurring questions and concepts
Add examination-year trend analysis
Improve prediction accuracy using larger datasets
Add interactive data visualizations


👨‍💻 Author

Bennu Durgesh Naraharisetti

GitHub: https://github.com/Durgesh177
LinkedIn: https://www.linkedin.com/in/bennu-durgesh-naraharisetti-819812352