from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import zipfile
import os
import os
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as pltpip
from nltk.stem import PorterStemmer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import rbf_kernel

import joblib

app = Flask(__name__)
CORS(app)
directory = "extracted_files"
stop_words_file = "stopwords.txt"
stop_directory = "stop_words_removed"
stem_directory = "AFTER_STEMMING"
excel_file_name = "Frequency_representation.csv"
cats_path = "cats.txt"
docs = None
text_docs = None
selectedalgo = ""
item = ""
bal_acc = 0
acc = 0
precision = 0
f1_score = 0
recall = 0


def load_documents():
    global docs, text_docs
    if docs is None or text_docs is None:
        docs = []
        text_docs = [f for f in os.listdir(directory)]
        for file_name in text_docs:
            with open(
                os.path.join(directory, file_name), "r", encoding="ISO-8859-1"
            ) as file:
                content = file.read()
                docs.append(content)
    return docs, text_docs


def ig(labels, binary_arr, vectorizer):
    rows = binary_arr.shape[0]
    cols = binary_arr.shape[1]
    total_ig = {}
    for i in labels:
        if i in total_ig:
            total_ig[i] += 1
        else:
            total_ig[i] = 1
    lis = total_ig.values()
    
@app.route("/remove_stopwords", methods=["GET"])
def remove_stopwords():
    global docs, text_docs
    docs, text_docs = load_documents()
    stop_words = []
    # reading stop words from txt file
    with open(stop_words_file, "r") as file:
        for line in file:
            stop_words.append(line.strip())

    # Create the stop_words_removed directory if it doesn't exist
    if not os.path.exists(stop_directory):
        os.makedirs(stop_directory)

    # stopwords removal
    for i in range(len(docs)):
        text = docs[i].split()
        docs[i] = " ".join(word for word in text if word.lower() not in stop_words)
        with open(os.path.join(stop_directory, text_docs[i]), "w") as file:
            file.write(docs[i])

    return jsonify(
        {
            "success": True,
            "message": "Stopwords removed",
        }
    )


@app.route("/perform_stemming", methods=["GET"])
def perform_stemming():
    global docs, text_docs
    docs, text_docs = load_documents()
    stemmer = PorterStemmer()
    # stemming
    for i in range(len(docs)):
        text = docs[i].split()
        content = []
        for word in text:
            if word.isalpha():
                content.append(stemmer.stem(word))
        docs[i] = " ".join(content)
        if not os.path.exists(stem_directory):
            os.makedirs(stem_directory)
        with open(os.path.join(stem_directory, text_docs[i]), "w") as file:
            file.write(docs[i])

    return jsonify(
        {
            "success": True,
            "message": "Stemming completed",
        }
    )


@app.route("/generate_representation", methods=["GET"])
def generate_representation():
    global docs, text_docs
    docs, text_docs = load_documents()


    vectorizer = CountVectorizer()
    Freq_rep = vectorizer.fit_transform(docs)
    data = Freq_rep.toarray()
    a = vectorizer.get_feature_names_out()
    b = text_docs
    for i in range(len(b)):
        b[i] = int(b[i])
    Binary_rep = vectorizer.fit_transform(docs)
    non_zero_mask = Binary_rep != 0
    Binary_rep[non_zero_mask] = 1

    # FREQUENCY REPRESENTATION
    df = pd.DataFrame(data, columns=a, index=b)
    df = df.sort_index()

    #Extracting labels
    labels = []
    i = 1
    with open(cats_path, 'r', encoding = 'ISO-8859-1') as file:
        for line in file:
            if i == 714:
                break
            c = line.split(':')[1][:-1]
            labels.append(c)
            i += 1
    
    # BINARY REPRESENTATION
    Binary_rep = vectorizer.fit_transform(docs)
    non_zero_mask = Binary_rep != 0
    Binary_rep[non_zero_mask] = 1
    binary_arr = Binary_rep.toarray()
    bf = pd.DataFrame(binary_arr, columns=a,index=b)
    bf = df.sort_index()

    # Tuning
    test_attributes = set(df.columns)
    attributes = set()
    text_file = r".\Attributes.txt"
    with open(text_file,'r') as file:
        for line in file:
            attributes.add(line[:-1])
    drop_cols = (test_attributes) - attributes 
    df.drop(drop_cols,axis=1,inplace=True)
    bf.drop(drop_cols,axis=1,inplace=True)
    add_cols = attributes - test_attributes 
    add_cols = list(add_cols)
    for i in add_cols:
        df [i] = 0
    for i in add_cols:
        bf [i] = 0
    
    sorted_columns = sorted(df.columns)
    df = df[sorted_columns]
    df['Label'] = labels

    sorted_columns = sorted(bf.columns)
    bf = bf[sorted_columns]
    bf['Label'] = labels

    bf.to_csv("Binary_representation.csv")
    df.to_csv("Frequency_representation.csv")

    return jsonify(
        {
            "success": True,
            "message": "Frequency representation generated",
        }
    )


@app.route("/upload_zip", methods=["POST"])
def upload_zip():
    file = request.files["file"]
    if file and file.filename.endswith(".zip"):
        extract_folder = "extracted_files"
        os.makedirs(extract_folder, exist_ok=True)
        zip_path = os.path.join(extract_folder, file.filename)
        file.save(zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)
        extracted_files = os.listdir(extract_folder)

        os.remove(zip_path)
        return jsonify(
            {"success": True, "message": "Zip file uploaded and extracted successfully"}
        )
    else:
        return jsonify(
            {"success": False, "message": "Invalid file type. Please upload a zip file"}
        )


@app.route("/get_extracted_files", methods=["GET"])
def get_extracted_files():
    try:
        extract_folder = "extracted_files"

        # Check if the directory exists
        if not os.path.exists(extract_folder):
            return jsonify({"success": False, "message": "Directory not found"})

        # Get the list of files in the directory
        file_list = os.listdir(extract_folder)

        return jsonify(
            {
                "success": True,
                "message": "Files retrieved successfully",
                "num_files": len(file_list),
                "file_names": file_list,
            }
        )
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"success": False, "message": "Internal server error"})


@app.route("/runalgo", methods=["POST"])
def runalgo():
    global selectedalgo, item, acc, bal_acc, precision, f1_score, recall

    dt = pd.read_csv("./Frequency_representation.csv", index_col=0)
    Y = dt["Label"]
    dt.drop("Label", axis=1, inplace=True)
    print(dt.shape)

    data_req = request.json
    if data_req:
        selectedalgo = data_req.get("selectedalgo")
        item = data_req.get("item")
        print(selectedalgo, item)


        #DECISION TREE
        if selectedalgo == "Decision Tree" and item == "With_Descretization":
            model = joblib.load("./Models/Decision_Tree/dt_desc_split1.pkl")

            #Descretizing the data
            data_ = np.ascontiguousarray(dt)
            data1 = pd.DataFrame()
            bins = np.array([-1,0,1,2,4,8,10,17,22,49], dtype=np.int64)
            labels = np.array([1,2,3,4,5,6,7,8,9], dtype=np.int64)
            for i in range(dt.shape[1]):
                data1[i] = pd.cut(data_[:, i], bins=bins, labels=labels)

            X = data1
            y_pred = model.predict(X)
            report = classification_report(y_pred, Y, output_dict=True)
            precision = report['weighted avg']['precision']
            f1_score = report['weighted avg']['f1-score']
            recall = report['weighted avg']['recall']
            acc = accuracy_score(Y, y_pred)
            bal_acc = balanced_accuracy_score(Y, y_pred)
            print(acc, bal_acc)
            return (
                jsonify(
                    {
                        "message": "Data received and processed successfully.",
                        "data": {
                            "item": item,
                            "selectedalgo": selectedalgo,
                            "acc" : round(acc*100,2),
                            "bal_acc": round(bal_acc * 100, 2),
                            "precision" : round(precision, 2),
                            "f1_score" : round(f1_score, 2),
                            "recall" : round(recall, 2)
                        },
                    }
                ),
                200,
            )
        elif selectedalgo == "Decision Tree" and item == "Without_Descretization":
            model = joblib.load("./Models/Decision_Tree/dt_without_disc.pkl")
            print("predic")
            X = dt.copy()
            print(X.shape)
            y_pred = model.predict(X)
            report = classification_report(y_pred, Y, output_dict=True)
            print("predicting")
            precision = report['weighted avg']['precision']
            f1_score = report['weighted avg']['f1-score']
            recall = report['weighted avg']['recall']
            acc = accuracy_score(Y, y_pred)
            bal_acc = balanced_accuracy_score(Y, y_pred)


            return (
                jsonify(
                    {
                        "message": "Data received and processed successfully.",
                        "data": {
                            "item": item,
                            "selectedalgo": selectedalgo,
                            "acc" : round(acc*100,2),
                            "bal_acc": round(bal_acc * 100, 2),
                            "precision" : round(precision, 2),
                            "f1_score" : round(f1_score, 2),
                            "recall" : round(recall, 2)
                        },
                    }
                ),
                200,
            )
        elif selectedalgo == "Decision Tree" and item == "Gaussian":
            model = joblib.load('./Models/decision_Tree/DT_R8-Guassian.pkl')

            r8_df = pd.read_csv("R8_frequency_rep.csv",index_col=0)
            r8_df = r8_df.sort_index()

            print('predict')
            labels = r8_df['Label']
            r8_df.drop('Label',axis=1,inplace=True)

            gam = 1 / r8_df.shape[1]
            similarity_matrix = rbf_kernel(r8_df, gamma=gam)

            X = similarity_matrix.copy()
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=42)

            print('prediction')
            y_pred = model.predict(X_test)
            report = classification_report(y_pred, y_test, output_dict=True)

            precision = report['weighted avg']['precision']
            f1_score = report['weighted avg']['f1-score']
            recall = report['weighted avg']['recall']
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)

            return (
                jsonify(
                    {
                        "message": "Data received and processed successfully.",
                        "data": {
                            "item": item,
                            "selectedalgo": selectedalgo,
                            "acc" : round(acc*100,2),
                            "bal_acc": round(bal_acc * 100, 2),
                            "precision" : round(precision, 2),
                            "f1_score" : round(f1_score, 2),
                            "recall" : round(recall, 2)
                        },
                    }
                ),
                200,
            )
        
        #LOGISTIC REGRESSION
        elif selectedalgo == "Logistic Regression" and item == "Constant Threshold":
            model = joblib.load("./Models/Logistic_Regression/LR-R8-CT.pkl")

            X = dt.copy()
            y_pred = model.predict(X)

            report = classification_report(y_pred, Y, output_dict=True)
            precision = report['weighted avg']['precision']
            f1_score = report['weighted avg']['f1-score']
            recall = report['weighted avg']['recall']
            acc = accuracy_score(Y, y_pred)
            bal_acc = balanced_accuracy_score(Y, y_pred)

            return (
                jsonify(
                    {
                        "message": "Data received and processed successfully.",
                        "data": {
                            "item": item,
                            "acc" : round(acc*100,2),
                            "selectedalgo": selectedalgo,
                            "bal_acc": round(bal_acc * 100, 2),
                            "precision" : round(precision, 2),
                            "f1_score" : round(f1_score, 2),
                            "recall" : round(recall, 2)
                        },
                    }
                ),
                200,
            )

        elif selectedalgo == "KNN" and item == "Euclidean":
            return (
                jsonify(
                    {
                        "message": "Data received and processed successfully.",
                        "data": {
                            "item": item,
                            "selectedalgo": selectedalgo,
                        },
                    }
                ),
                200,
            )
        else:
            return jsonify({"error": "No data found in the request body."}), 400


@app.route("/get_results", methods=["GET"])
def get_results():
    data = {
        "item": item,
        "selectedalgo": selectedalgo,
        "acc": round(acc * 100, 2),
        "bal_acc": round(bal_acc * 100, 2),
        "precision" : round(precision, 2),
        "f1_score" : round(f1_score, 2),
        "recall" : round(recall, 2)

    }
    return (
        jsonify({"message": "Data received and processed successfully.", "data": data}),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True)
