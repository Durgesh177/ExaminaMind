from datetime import datetime
import os
import json

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    send_from_directory
)

from utils import extract_text, preprocess
from model import train_model, predict


# ---------------------------------
# FLASK APP SETUP
# ---------------------------------

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Store extracted question-paper text
stored_texts = []


# Track whether the model has been trained
model_trained = False


# ---------------------------------
# DASHBOARD
# ---------------------------------

@app.route("/")
def dashboard():

    files = []

    for filename in os.listdir(UPLOAD_FOLDER):

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        if os.path.isfile(filepath):

            modified_time = datetime.fromtimestamp(
                os.path.getmtime(filepath)
            )

            files.append({
                "name": filename,
                "time": modified_time.strftime(
                    "%Y-%m-%d %H:%M"
                )
            })

    return render_template(
        "dashboard.html",
        files=files,
        count=len(files)
    )


# ---------------------------------
# UPLOAD FILE
# ---------------------------------

@app.route("/upload", methods=["POST"])
def upload():

    global model_trained

    file = request.files.get("file")

    if not file or not file.filename:

        return "No file selected."

    allowed_extensions = {
        ".pdf",
        ".docx",
        ".txt"
    }

    file_extension = os.path.splitext(
        file.filename
    )[1].lower()

    if file_extension not in allowed_extensions:

        return (
            "Unsupported file format. "
            "Please upload PDF, DOCX, or TXT files."
        )

    try:

        filepath = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        file.save(filepath)

        print("✅ File saved:", filepath)

        # Extract text
        text = extract_text(filepath)

        print(
            "Extracted text length:",
            len(text)
        )

        if not text.strip():

            return (
                "Could not extract text "
                "from the uploaded file."
            )

        # Preprocess text
        processed_text = preprocess(text)

        # Use processed text if available
        final_text = (
            processed_text
            if processed_text.strip()
            else text
        )

        # Store text for training
        stored_texts.append(
            final_text
        )

        # New upload means
        # the existing model should be retrained
        model_trained = False

        return redirect("/analysis")

    except Exception as error:

        print(
            "❌ Upload error:",
            error
        )

        return str(error)


# ---------------------------------
# ANALYSIS PAGE
# ---------------------------------

@app.route("/analysis")
def analysis():

    return render_template(
        "analysis.html",
        count=len(stored_texts)
    )


# ---------------------------------
# TRAIN MODEL
# ---------------------------------

@app.route("/train")
def train():

    global model_trained

    if len(stored_texts) < 2:

        return (
            "Please upload at least "
            "two question papers before training."
        )

    model_trained = train_model(
        stored_texts
    )

    if not model_trained:

        return (
            "Model training failed. "
            "Try uploading question papers "
            "from different topics."
        )

    return redirect(
        "/prediction"
    )


# ---------------------------------
# PREDICTION PAGE
# ---------------------------------

@app.route("/prediction")
def prediction_page():

    if not stored_texts:

        return render_template(
            "prediction.html",
            results=[],
            accuracy=0
        )

    # Load prediction results
    # from the most recently uploaded paper
    results = predict(
        stored_texts[-1]
    )

    if not results:

        return render_template(
            "prediction.html",
            results=[],
            accuracy=0
        )

    # Highest probability
    confidence = max(
        probability
        for _, probability
        in results
    )

    accuracy = round(
        confidence * 100,
        2
    )

    return render_template(
        "prediction.html",
        results=results,
        accuracy=accuracy
    )


# ---------------------------------
# VISUALIZATION PAGE
# ---------------------------------

@app.route("/visualization")
def visualization_page():

    if not stored_texts:

        return render_template(
            "visualization.html",
            results=[]
        )

    results = predict(
        stored_texts[-1]
    )

    return render_template(
        "visualization.html",
        results=json.dumps(results)
    )


# ---------------------------------
# DOWNLOAD FILE
# ---------------------------------

@app.route("/download/<filename>")
def download_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename,
        as_attachment=True
    )


# ---------------------------------
# DELETE FILE
# ---------------------------------

@app.route("/delete/<filename>")
def delete_file(filename):

    global model_trained

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    if os.path.exists(filepath):

        os.remove(filepath)

    # Model may need retraining
    model_trained = False

    return redirect("/")


# ---------------------------------
# RUN APPLICATION
# ---------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )


Code Developed By Bennu Durgesh N
                  Tarun Varma G
