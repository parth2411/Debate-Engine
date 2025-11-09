# 🥊 AI vs. AI Debate Arena

**AI vs. AI Debate Arena** is a Streamlit-based application that allows multiple large language models (LLMs) — such as **Groq**, **Google Gemini**, and others — to debate against each other on any topic you choose.  
A third AI model acts as the **judge**, evaluating clarity, coherence, and persuasiveness to declare a winner.

The app stores debate history on **AWS S3**, supports **dynamic model selection**, and visualizes **round-by-round performance** with interactive charts.


## 🚀 Features

- 🧠 **Interactive Streamlit UI** — Simple, elegant interface for running debates  
- ⚔️ **Multi-Model Support** — Choose from various LLMs (Gemini, LLaMA, DeepSeek, Mistral, Qwen, etc.)  
- ⚖️ **AI Judge Evaluation** — Automatically scores debates using weighted criteria  
- ☁️ **Automatic S3 Integration** — Debate history is securely stored and downloadable with a password  
- 🔄 **Customizable Debate Flow** — Configure number of rounds, model selection, and debater visibility  
- 📊 **Performance Visualization** — Bar chart summary of debate scores across rounds  
- 🪶 **Error Handling & Logging** — Robust logging for API calls and S3 uploads  


## 🧠 System Architecture

```

[User Input → Streamlit UI]
↓
[Groq / Gemini API Calls]
↓
[Debater A ↔ Debater B]
↓
[Judge AI]
↓
[Evaluation + Score Extraction]
↓
[S3 Upload + Visualization]

````


## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/ai-debate-arena.git
cd ai-debate-arena
````

### 2️⃣ Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # For macOS/Linux
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Your `requirements.txt` should include:

```
streamlit
boto3
groq
google-generativeai
python-dotenv
```

### 4️⃣ Set Up Environment Variables

Create a `.env` file in the root directory with your credentials:

```
# API Keys
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key

# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
S3_BUCKET=your_s3_bucket_name

# Optional Password for Downloads
DOWNLOAD_PASSWORD=your_secure_password
```

### 5️⃣ Run the App

```bash
streamlit run app.py
```

Then open your browser at:
👉 [http://localhost:8501](http://localhost:8501)


## 🧩 How It Works

1. **Choose a topic** — Enter a motion (e.g., “AI will replace most jobs”).
2. **Select models** for:

   * Debater A (Pro)
   * Debater B (Con)
   * Judge
3. **Start the debate** — Models take turns presenting:

   * Opening statements
   * Multi-round arguments
   * Cross-examinations
   * Closing statements
4. **Judge evaluates** each round based on:

   * Clarity (40%)
   * Coherence (40%)
   * Persuasiveness (20%)
5. **Final winner is declared** — Results are uploaded to S3 and visualized.


## 📊 Example Output

**Judge Evaluation Example:**

```
Score A: 7.50 (Clarity: 7.80, Coherence: 7.60, Persuasiveness: 7.00)
Score B: 6.80 (Clarity: 6.90, Coherence: 6.70, Persuasiveness: 6.80)
Explanation: Argument A is clearer and more coherent overall.
```

**Visualization Example:**

* Interactive bar chart showing Debater A vs. Debater B scores for each debate stage.


## ☁️ AWS S3 Integration

All debate sessions are appended to a file named `debate_history.json` in your configured S3 bucket.

You can also:

* **Download debate history** via password-protected access
* **Review previous sessions** directly in the Streamlit “Debate History” section

---

## 🧰 Key Functions

| Function                                | Description                                          |
| --------------------------------------- | ---------------------------------------------------- |
| `generate_opening()`                    | Creates an opening statement for a debater           |
| `generate_argument()`                   | Generates a structured argument for each round       |
| `cross_examine()`                       | Identifies flaws in the opponent’s argument          |
| `evaluate_argument()`                   | Scores and explains the judge’s decision             |
| `extract_scores()`                      | Parses structured judge feedback into numeric scores |
| `upload_to_s3()` / `download_from_s3()` | Handles debate data persistence                      |


## 🧱 Project Structure

```
ai-debate-arena/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
├── .env                   # Environment variables (not committed)
└── README.md              # Documentation
```


## 🔒 Security Notes

* Never commit your `.env` file or API keys
* Use IAM policies to restrict S3 bucket access
* Change `DOWNLOAD_PASSWORD` periodically


## 🤖 Supported Models

### Groq Models

* llama3-70b-8192
* deepseek-r1-distill-llama-70b
* mistral-saba-24b
* qwen-qwq-32b
* meta-llama/llama-4-maverick-17b-128e-instruct
* gemma2-9b-it

### Google Gemini Models

* gemini-1.5-pro
* gemini-2.0-flash
* gemini-2.5-flash-preview-05-20
* gemini-2.5-pro-preview-05-06
* gemini-2.0-flash-lite


## 💡 Future Enhancements

* 🎙️ Add voice-based debates using Whisper AI
* 💬 Real-time chat display with markdown formatting
* 🧮 Multi-judge evaluation and average scoring
* ⚙️ Enable user-uploaded models via local endpoints


## 🧑‍💻 Author

**Parth Bhalodiya**
Senior AI/ML Engineer
📫 [LinkedIn](https://www.linkedin.com/in/parth-bhalodiya-555011128) • [GitHub](https://github.com/parth2411)


## 📜 License

This project is licensed under the **MIT License**.
Feel free to use, modify, and share it responsibly.


