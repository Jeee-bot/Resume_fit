
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import json

import docx2txt
from sklearn.feature_extraction.text import CountVectorizer
# import sklearn to use the cosine similarity function
from sklearn.metrics.pairwise import cosine_similarity

ALLOWED_EXTENSIONS = set(['doc', 'docx'])

app = Flask(__name__)  # Create an instance of a web application
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
CORS(app)  # Enable


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cv_scorer(job_description, cv):

    pred_score = {} # Initialize dictionary

    if job_description.lower() == "data scientist":
        jd_file = "Data Scientist_JD.docx"
    elif job_description.lower() == "content strategist":
        jd_file = "Content_Strategist.docx"

    JD = docx2txt.process(jd_file)

    resume = docx2txt.process(cv)

    # create a list of text
    text=[resume, JD]

    cv = CountVectorizer()
    count_matrix =cv.fit_transform(text)

    # Append similarity scores
    pred_score["similarity_score"] = cosine_similarity(count_matrix)

    # get the match percentage
    matchPercentage = cosine_similarity(count_matrix)[0][1]*100
    matchPercentage = round(matchPercentage, 2) # round to 2 decimal places

    # Append Resume Match Percentage
    pred_score["resume_match_percentage"] = matchPercentage

    return pred_score

@app.route('/cv-scorer', methods=['POST'])  # Decorator for the function
def get_input():
    text = request.form['text'] # Read text
    # check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message': 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        resp = jsonify({'message': 'File successfully uploaded'})
        resp.status_code = 201
        d = cv_scorer(text, file)
        return json.dumps({k: v.tolist() for k, v in d.items()})
    else:
        resp = jsonify({'message': 'Allowed file types are doc, docx'})
        resp.status_code = 400
        return resp


if __name__ == "__main__":
    # app.run(debug=True) # Only while developing code
    app.run()  # Run the app
