from flask import Flask, render_template, request, jsonify
import tempfile
from upload_file import (
    load_pdf_file,
    group_docs_by_path,
    chunk_all_sections,
    push_to_vector_db,
    create_index
)
from query_engine import get_answer


app = Flask(__name__, template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/create_index", methods=["GET"])
def create_index():
    create_index()
    return jsonify({"body": "Index Created"})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or not file.filename.endswith(".pdf"):
        return jsonify({"error": "Invalid file"}), 400

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        file.save(temp.name)
        file_path = temp.name

    try:
        # Step 1: Load and extract
        document = load_pdf_file(file_path)
        doc_map = group_docs_by_path(document)

        # Step 2: Chunk and index
        doc_map = chunk_all_sections(doc_map)
        push_to_vector_db(doc_map)

        return jsonify({"message": "Uploaded and indexed successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["GET"])
def query():
    question = request.args.get("question")
    if not question:
        return jsonify({"error": "Missing question parameter"}), 400

    try:
        result = get_answer(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8001)