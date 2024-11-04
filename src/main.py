from flask import Flask, render_template, request

app = Flask(__name__)

from src.db.chroma_client import get_answer_from_db

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    user_question = None
    original_question = None  # Initialize with default value
    similarity = None  # Initialize with default value

    if request.method == "POST":
        user_question = request.form.get("question")
        print("**********************************")
        print(user_question)
        print("*********************")

        # Ensure `get_answer_from_db` returns a result before accessing indices
        result = get_answer_from_db(user_question, similarity_threshold=0.7)
        
        if result:  # Check if result is not None or empty
            answer = result[0]
            original_question = result[1]
            similarity = result[2]

    return render_template("index.html", question=user_question, answer=answer, original_question=original_question, similarity=similarity)

if __name__ == "__main__":
    app.run(debug=True)
