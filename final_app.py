import streamlit as st
import json
import boto3
from botocore.exceptions import ClientError
from groq import Groq
import google.generativeai as genai
import uuid
import re
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize APIs with error handling
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    st.error(f"Failed to initialize APIs: {str(e)}")
    st.stop()

# Initialize AWS S3 client
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )
except Exception as e:
    st.error(f"Failed to initialize AWS S3 client: {str(e)}")
    st.stop()

# Get S3 bucket and download password from environment variables
S3_BUCKET = os.getenv("S3_BUCKET")
DOWNLOAD_PASSWORD = os.getenv("DOWNLOAD_PASSWORD")
json_filename = "debate_history.json"

# Available models for selection
MODEL_OPTIONS = {
    "Select a model": None,  # Placeholder option
    "Gemini 2.0 Flash": "gemini-2.0-flash",
    "Gemini 1.5 Pro": "gemini-1.5-pro",
    "Gemini 1.5 flash": "gemini-1.5-flash",
    "Gemini 2.5 flash": "gemini-2.5-flash-preview-05-20",
    "Gemini 2.5 pro": "gemini-2.5-pro-preview-05-06",
    "LLaMA3 70B": "llama3-70b-8192",
    "Deepseek r1": "deepseek-r1-distill-llama-70b",
    "Gemini 2.0 Flash lite": "gemini-2.0-flash-lite",
    "Gemma 9b": "gemma2-9b-it",
    "Mistral saba 24b": "mistral-saba-24b",
    "Qwen 32b": "qwen-qwq-32b",
    "LLaMA4 17b": "meta-llama/llama-4-maverick-17b-128e-instruct",
}

# Supported model prefixes for validation
GROQ_PREFIXES = ["llama3", "deepseek", "mistral", "qwen", "meta", "gemma"]
GEMINI_PREFIXES = ["gemini"]

# A2A Agent Cards
debater_a_card = {
    "agent_id": str(uuid.uuid4()),
    "name": "Debater A (Pro)",
    "capabilities": ["generate_argument", "cross_examine", "opening_statement", "closing_statement", "answer_question"],
    "endpoint": "http://localhost:8000/debater_a",
    "auth": "none"
}
debater_b_card = {
    "agent_id": str(uuid.uuid4()),
    "name": "Debater B (Con)",
    "capabilities": ["generate_argument", "cross_examine", "opening_statement", "closing_statement", "answer_question"],
    "endpoint": "http://localhost:8000/debater_b",
    "auth": "none"
}
judge_card = {
    "agent_id": str(uuid.uuid4()),
    "name": "Judge",
    "capabilities": ["evaluate_argument", "declare_winner"],
    "endpoint": "http://localhost:8000/judge",
    "auth": "none"
}

# Define team labels at top scope
team1_label = None
team2_label = None

def validate_model(model_name, api_prefixes):
    """Validate if the model name is supported by the specified API."""
    return model_name and any(model_name.startswith(prefix) for prefix in api_prefixes)

def a2a_communicate(agent_card, task, payload):
    """Simulate A2A communication with error handling."""
    model_name = agent_card.get("model")
    try:
        if validate_model(model_name, GROQ_PREFIXES):
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": f"{task}: {json.dumps(payload)}"}],
                model=model_name,
                temperature=0.5
            )
            return response.choices[0].message.content
        elif validate_model(model_name, GEMINI_PREFIXES):
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(f"{task}: {json.dumps(payload)}")
            return response.text
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    except Exception as e:
        logger.error(f"Error in A2A communication for {agent_card['name']} with model {model_name}: {str(e)}")
        return f"Error: Unable to generate response due to {str(e)}"

def generate_opening(debater_card, topic, stance):
    """Generate a concise opening statement."""
    prompt = f"Debate topic: {topic}. Stance: {stance}. Provide a concise opening statement (max 100 words) outlining your position."
    if debater_card["name"] == "Debater B (Con)":
        prompt += " Argue confidently, emphasizing key counterpoints."
    return a2a_communicate(debater_card, "opening_statement", {"topic": topic, "stance": stance})

def generate_argument(debater_card, topic, stance, round_num):
    """Generate a concise argument for the given round."""
    prompt = f"Debate topic: {topic}. Stance: {stance}. Round {round_num}. Generate a concise argument (max 250 words) supporting your position."
    if debater_card["name"] == "Debater B (Con)":
        prompt += " Argue confidently, emphasizing strong counterpoints."
    return a2a_communicate(debater_card, "generate_argument", {"topic": topic, "stance": stance, "round": round_num})

def cross_examine(debater_card, opponent_argument):
    """Cross-examine the opponent's argument."""
    prompt = f"Cross-examine this argument: {opponent_argument}. Identify one specific flaw or inconsistency (max 100 words)."
    return a2a_communicate(debater_card, "cross_examine", {"argument": opponent_argument})

def generate_closing(debater_card, topic, stance, history):
    """Generate a concise closing statement."""
    prompt = f"Debate topic: {topic}. Stance: {stance}. Summarize your key arguments and rebuttals in a concise closing statement (max 150 words). History: {json.dumps(history)}"
    if debater_card["name"] == "Debater B (Con)":
        prompt += " Reinforce your position confidently to win."
    return a2a_communicate(debater_card, "closing_statement", {"topic": topic, "stance": stance, "history": history})

def evaluate_argument(judge_card, argument_a, argument_b, round_num, stage="Primary Argument"):
    """Evaluate arguments with rigorous, weighted criteria."""
    prompt = f"""Evaluate {stage} for Round {round_num} using:
- Clarity (40%): Structure, precision, ease of understanding (0.00-10.00).
- Coherence (40%): Logic, consistency, relevance (0.00-10.00).
- Persuasiveness (20%): Appeal, evidence, impact (0.00-10.00).
Argument A: {argument_a}
Argument B: {argument_b}
Calculate total: (Clarity * 0.4) + (Coherence * 0.4) + (Persuasiveness * 0.2).
Return exactly in this format:
```
Score A: X.XX (Clarity: Y.YY, Coherence: Z.ZZ, Persuasiveness: W.WW)
Score B: P.PP (Clarity: Q.QQ, Coherence: R.RR, Persuasiveness: S.SS)
Explanation: [Detailed analysis, max 150 words]
```
Assign distinct scores reflecting argument quality. If one argument is stronger, its score must be higher. Example:
```
Score A: 7.50 (Clarity: 7.80, Coherence: 7.60, Persuasiveness: 7.00)
Score B: 6.80 (Clarity: 6.90, Coherence: 6.70, Persuasiveness: 6.80)
Explanation: Argument A is clearer and more coherent...
```"""
    return a2a_communicate(judge_card, "evaluate_argument", {"arg_a": argument_a, "arg_b": argument_b})

def extract_scores(evaluation):
    """Extract scores from judge's evaluation or infer from text if format is missing."""
    pattern = r"Score A: (\d+\.\d{2}) \(Clarity: (\d+\.\d{2}), Coherence: (\d+\.\d{2}), Persuasiveness: (\d+\.\d{2})\)\s*Score B: (\d+\.\d{2}) \(Clarity: (\d+\.\d{2}), Coherence: (\d+\.\d{2}), Persuasiveness: (\d+\.\d{2})\)"
    match = re.search(pattern, evaluation)
    if match:
        total_a, clarity_a, coherence_a, persuasiveness_a, total_b, clarity_b, coherence_b, persuasiveness_b = map(float, match.groups())
        return total_a, total_b, persuasiveness_a, persuasiveness_b
    
    logger.warning(f"Score extraction failed for evaluation: {evaluation}")
    evaluation_lower = evaluation.lower()
    b_stronger_phrases = [
        "argument b is stronger", "argument b presents a stronger", "argument b is more persuasive",
        "argument b effectively counters", "argument b is more compelling", "argument b provides a stronger",
        "argument b presents a more"
    ]
    a_stronger_phrases = [
        "argument a is stronger", "argument a presents a stronger", "argument a is more persuasive",
        "argument a effectively counters", "argument a is more compelling", "argument a provides a stronger",
        "argument a presents a more"
    ]
    
    if any(phrase in evaluation_lower for phrase in b_stronger_phrases):
        return 6.5, 7.5, 6.5, 7.5
    elif any(phrase in evaluation_lower for phrase in a_stronger_phrases):
        return 7.5, 6.5, 7.5, 6.5
    else:
        return 5.0, 5.0, 5.0, 5.0

def upload_to_s3(data, bucket, key):
    """Upload JSON data to S3."""
    try:
        json_data = json.dumps(data, indent=4)
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json_data.encode('utf-8'),
            ContentType='application/json'
        )
        logger.info(f"Successfully uploaded {key} to S3 bucket {bucket}")
        return True
    except ClientError as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        return False

def download_from_s3(bucket, key):
    """Download JSON file from S3."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()
    except ClientError as e:
        logger.error(f"Error downloading from S3: {str(e)}")
        return None

# Custom CSS
st.markdown("""
<style>
.debate-container {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
}
.debate-title {
    text-align: center;
    font-size: 2.5em;
    color: #1a1a1a;
}
.debate-stage {
    display: flex;
    justify-content: space-between;
}
.debate-column {
    width: 45%;
    padding: 10px;
    border-radius: 5px;
}
.debate-a {
    background-color: #e6f3ff;
    border: 2px solid #0000FF;
}
.debate-b {
    background-color: #ffe6e6;
    border: 2px solid #FF0000;
}
.judge-section {
    background-color: #e6ffe6;
    border: 2px solid #008000;
    padding: 10px;
    border-radius: 5px;
    text-align: center;
}
.section-divider {
    margin: 20px 0;
    padding-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown("<div class='debate-title'>AI vs. AI Debate Arena</div>", unsafe_allow_html=True)

topic = st.text_input("Debate Motion", placeholder="Enter the debate motion")
num_rounds = st.slider("Number of Argument Rounds", 1, 5, 3)
hide_debaters = st.checkbox("Hide Debater Identities (Show as Team 1 & Team 2)", value=False)

st.subheader("Model Selection")
col1, col2, col3 = st.columns(3)
with col1:
    debater_a_model = st.selectbox("Debater A Model", list(MODEL_OPTIONS.keys()), index=None, placeholder="Select a model")
    debater_a_card["model"] = MODEL_OPTIONS[debater_a_model] if debater_a_model and debater_a_model != "Select a model" else None
with col2:
    debater_b_model = st.selectbox("Debater B Model", list(MODEL_OPTIONS.keys()), index=None, placeholder="Select a model")
    debater_b_card["model"] = MODEL_OPTIONS[debater_b_model] if debater_b_model and debater_b_model != "Select a model" else None
with col3:
    judge_model = st.selectbox("Judge Model", list(MODEL_OPTIONS.keys()), index=None, placeholder="Select a model")
    judge_card["model"] = MODEL_OPTIONS[judge_model] if judge_model and judge_model != "Select a model" else None

start_debate = st.button("Start Debate 🥊")

# Validate inputs before starting debate
if start_debate:
    errors = []
    if not topic.strip():
        errors.append("Debate motion is required.")
    if debater_a_card["model"] is None:
        errors.append("Debater A model must be selected.")
    if debater_b_card["model"] is None:
        errors.append("Debater B model must be selected.")
    if judge_card["model"] is None:
        errors.append("Judge model must be selected.")
    
    for agent, model in [("Debater A", debater_a_card["model"]), ("Debater B", debater_b_card["model"]), ("Judge", judge_card["model"])]:
        if model and not (validate_model(model, GROQ_PREFIXES) or validate_model(model, GEMINI_PREFIXES)):
            errors.append(f"Invalid model selected for {agent}: {model}.")
    
    if errors:
        for error in errors:
            st.error(error)
    else:
        team1_label = "Team 1" if hide_debaters else f"Debater A (Pro, {debater_a_model})"
        team2_label = "Team 2" if hide_debaters else f"Debater B (Con, {debater_b_model})"
        
        debate_data = {
            "debate_id": str(uuid.uuid4()),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "topic": topic,
            "debater_a_model": debater_a_model,
            "debater_b_model": debater_b_model,
            "judge_model": judge_model,
            "rounds": [],
            "final_result": {}
        }
        
        st.session_state.chat_history = []
        st.session_state.scores_a = []
        st.session_state.scores_b = []
        
        st.markdown("<div class='debate-stage'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"<div class='debate-a'><strong>{team1_label}</strong></div>", unsafe_allow_html=True)
            with st.expander("Opening Statement"):
                opening_a = generate_opening(debater_a_card, topic, "Pro")
                st.markdown(opening_a, unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='debate-b'><strong>{team2_label}</strong></div>", unsafe_allow_html=True)
            with st.expander("Opening Statement"):
                opening_b = generate_opening(debater_b_card, topic, "Con")
                st.markdown(opening_b, unsafe_allow_html=True)
        
        evaluation_opening = evaluate_argument(judge_card, opening_a, opening_b, 0, "Opening Statement")
        with st.expander("Opening Evaluation"):
            st.markdown(f"<div class='judge-section'><strong>⚖️ Judge Evaluation: Opening Statements</strong><br>{evaluation_opening}</div>", unsafe_allow_html=True)
        score_a, score_b, _, _ = extract_scores(evaluation_opening)
        st.session_state.scores_a.append(score_a)
        st.session_state.scores_b.append(score_b)
        st.session_state.chat_history.append({
            "stage": "Opening",
            "arg_a": opening_a,
            "arg_b": opening_b,
            "evaluation": evaluation_opening
        })
        debate_data["rounds"].append({
            "stage": "Opening",
            "arg_a": opening_a,
            "arg_b": opening_b,
            "evaluation": evaluation_opening,
            "score_a": score_a,
            "score_b": score_b
        })
        
        for round_num in range(1, num_rounds + 1):
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            with col1:
                st.markdown(f"<div class='debate-a'><strong>{team1_label} (Round {round_num})</strong></div>", unsafe_allow_html=True)
                with st.expander(f"Argument (Round {round_num})"):
                    arg_a = generate_argument(debater_a_card, topic, "Pro", round_num)
                    st.markdown(arg_a, unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='debate-b'><strong>{team2_label} (Round {round_num})</strong></div>", unsafe_allow_html=True)
                with st.expander(f"Argument (Round {round_num})"):
                    arg_b = generate_argument(debater_b_card, topic, "Con", round_num)
                    st.markdown(arg_b, unsafe_allow_html=True)
            
            if round_num == num_rounds:
                st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                with col1:
                    st.markdown(f"<div class='debate-a'><strong>{team1_label} (Cross-Examination)</strong></div>", unsafe_allow_html=True)
                    with st.expander("Cross-Examination"):
                        cross_a = cross_examine(debater_a_card, arg_b)
                        st.markdown(cross_a, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='debate-b'><strong>{team2_label} (Cross-Examination)</strong></div>", unsafe_allow_html=True)
                    with st.expander("Cross-Examination"):
                        cross_b = cross_examine(debater_b_card, arg_a)
                        st.markdown(cross_b, unsafe_allow_html=True)
            
            evaluation_round = evaluate_argument(judge_card, arg_a, arg_b, round_num)
            with st.expander(f"Round {round_num} Evaluation"):
                st.markdown(f"<div class='judge-section'><strong>⚖️ Judge Evaluation: Round {round_num}</strong><br>{evaluation_round}</div>", unsafe_allow_html=True)
            score_a, score_b, _, _ = extract_scores(evaluation_round)
            st.session_state.scores_a.append(score_a)
            st.session_state.scores_b.append(score_b)
            st.session_state.chat_history.append({
                "stage": f"Round {round_num}",
                "arg_a": arg_a,
                "arg_b": arg_b,
                "evaluation": evaluation_round
            })
            debate_data["rounds"].append({
                "stage": f"Round {round_num}",
                "arg_a": arg_a,
                "arg_b": arg_b,
                "evaluation": evaluation_round,
                "score_a": score_a,
                "score_b": score_b
            })
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        with col1:
            st.markdown(f"<div class='debate-a'><strong>{team1_label}</strong></div>", unsafe_allow_html=True)
            with st.expander("Closing Statement"):
                closing_a = generate_closing(debater_a_card, topic, "Pro", st.session_state.chat_history)
                st.markdown(closing_a, unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='debate-b'><strong>{team2_label}</strong></div>", unsafe_allow_html=True)
            with st.expander("Closing Statement"):
                closing_b = generate_closing(debater_b_card, topic, "Con", st.session_state.chat_history)
                st.markdown(closing_b, unsafe_allow_html=True)
        
        evaluation_closing = evaluate_argument(judge_card, closing_a, closing_b, 0, "Closing Statement")
        with st.expander("Closing Evaluation"):
            st.markdown(f"<div class='judge-section'><strong>⚖️ Judge Evaluation: Closing Statements</strong><br>{evaluation_closing}</div>", unsafe_allow_html=True)
        score_a, score_b, persuasiveness_a, persuasiveness_b = extract_scores(evaluation_closing)
        st.session_state.scores_a.append(score_a)
        st.session_state.scores_b.append(score_b)
        st.session_state.chat_history.append({
            "stage": "Closing",
            "arg_a": closing_a,
            "arg_b": closing_b,
            "evaluation": evaluation_closing
        })
        debate_data["rounds"].append({
            "stage": "Closing",
            "arg_a": closing_a,
            "arg_b": closing_b,
            "evaluation": evaluation_closing,
            "score_a": score_a,
            "score_b": score_b
        })
        
        total_a = sum(st.session_state.scores_a)
        total_b = sum(st.session_state.scores_b)
        if total_a > total_b:
            winner = "Debater A (Pro) wins"
        elif total_b > total_a:
            winner = "Debater B (Con) wins"
        else:
            if persuasiveness_b > persuasiveness_a:
                winner = "Debater B (Con) wins (tiebreaker: higher persuasiveness in closing)"
            elif persuasiveness_a > persuasiveness_b:
                winner = "Debater A (Pro) wins (tiebreaker: higher persuasiveness in closing)"
            else:
                winner = "Debater A (Pro) wins (tiebreaker: evaluation text)" if "Argument B is stronger" not in evaluation_closing else "Debater B (Con) wins (tiebreaker: evaluation text)"
        debate_data["final_result"] = {
            "winner": winner,
            "total_score_a": total_a,
            "total_score_b": total_b
        }
        st.markdown(f"<div class='judge-section'><strong>Final Result: {winner}</strong> with scores {total_a:.2f} vs {total_b:.2f}</div>", unsafe_allow_html=True)
        
        # Append debate data to S3
        existing_data = []
        try:
            file_data = download_from_s3(S3_BUCKET, json_filename)
            if file_data:
                existing_data = json.loads(file_data.decode('utf-8'))
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                logger.error(f"Error reading from S3: {str(e)}")
                st.error(f"Error reading debate history from S3: {str(e)}")
        
        existing_data.append(debate_data)
        
        if upload_to_s3(existing_data, S3_BUCKET, json_filename):
            st.success(f"Debate data saved to S3 bucket {S3_BUCKET}")
        else:
            st.error("Failed to save debate data to S3. Check logs for details.")
        
        chart_data = {
            "type": "bar",
            "data": {
                "labels": ["Opening"] + [f"Round {i}" for i in range(1, num_rounds + 1)] + ["Closing"],
                "datasets": [
                    {"label": team1_label, "data": st.session_state.scores_a, "backgroundColor": "#0000FF"},
                    {"label": team2_label, "data": st.session_state.scores_b, "backgroundColor": "#FF0000"}
                ]
            },
            "options": {
                "scales": {"y": {"beginAtZero": True, "max": 10}},
                "plugins": {"title": {"display": True, "text": "Debate Scores by Stage"}}
            }
        }
        chart_html = f"""
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <canvas id="scoreChart" width="400" height="200"></canvas>
        <script>
            const ctx = document.getElementById('scoreChart').getContext('2d');
            new Chart(ctx, {json.dumps(chart_data)});
        </script>
        """
        st.components.v1.html(chart_html, height=300)
        st.markdown("</div>", unsafe_allow_html=True)

# Password-protected download section
st.subheader("Download Debate History")
password = st.text_input("Enter password to download debate history", type="password")
download_button = st.button("Download Debate History from S3")

if download_button and password:
    if password == DOWNLOAD_PASSWORD:
        file_data = download_from_s3(S3_BUCKET, json_filename)
        if file_data:
            st.download_button(
                label="Download Debate History",
                data=file_data,
                file_name=json_filename,
                mime="application/json"
            )
        else:
            st.error("Failed to download file from S3. Check logs for details.")
    else:
        st.error("Incorrect password.")

if "chat_history" in st.session_state:
    with st.expander("Debate History"):
        for entry in st.session_state.chat_history:
            st.write(f"**Stage: {entry['stage']}**")
            st.write(f"{team1_label}: {entry['arg_a']}")
            st.write(f"{team2_label}: {entry['arg_b']}")
            st.write(f"Judge: {entry['evaluation']}")
