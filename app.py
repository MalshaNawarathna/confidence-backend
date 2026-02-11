from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import tempfile
import spacy

nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

client = OpenAI()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    audio = request.files["audio"]
    print("Audio received")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.save(temp_file.name)

    with open(temp_file.name, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )

    text = transcription.text
    

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"""
Give confidence score (0-100) for this speech.
Return only number.

Speech:
{text}
"""}
        ]
    )

    score = result.choices[0].message.content.strip()

    doc = nlp(text)

    noun_count = 0
    verb_count = 0
    adj_count = 0
    total_words = 0

    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.pos_ == "NOUN":
                noun_count += 1
            elif token.pos_ == "VERB":
                verb_count += 1
            elif token.pos_ == "ADJ":
                adj_count += 1

    vocab_percentage = {
        "nouns": round(noun_count / total_words * 100, 2),
        "verbs": round(verb_count / total_words * 100, 2),
        "adjectives": round(adj_count / total_words * 100, 2)
    }

   

    return jsonify({"score": score,
                    "vocabulary": vocab_percentage})

if __name__ == "__main__":
    app.run(debug=True)

