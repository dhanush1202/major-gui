from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import zipfile
import os
import os
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
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
specificity1 = 0
y_pred = ['a'] * 713
file_dict = {}


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
        with open(os.path.join(stop_directory, str(text_docs[i])), "w") as file:
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
        with open(os.path.join(stem_directory, str(text_docs[i])), "w") as file:
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
    
   

    # Tuning
    test_attributes = set(df.columns)
    attributes = set()
    text_file = r".\Attributes.txt"
    with open(text_file,'r') as file:
        for line in file:
            attributes.add(line[:-1])
    drop_cols = (test_attributes) - attributes 
    df.drop(drop_cols,axis=1,inplace=True)
    add_cols = attributes - test_attributes 
    add_cols = list(add_cols)
    # for i in add_cols:
    #     df [i] = 0
    
    
    # sorted_columns = sorted(df.columns)
    # df = df[sorted_columns]
    # df['Label'] = labels

    additional_cols = pd.DataFrame(0, index=df.index, columns=add_cols)
    if not additional_cols.empty:
        df = pd.concat([df, additional_cols], axis=1)
    sorted_columns = sorted(df.columns)
    df = df[sorted_columns]
    df[ "Label" ] = labels

    # bf.to_csv("Binary_representation.csv")
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
    global selectedalgo, item, acc, bal_acc, precision, f1_score, recall, specificity1, y_pred, file_dict

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
            print(acc, bal_acc)
            
        elif selectedalgo == "Decision Tree" and item == "Without_Descretization":
            model = joblib.load("./Models/Decision_Tree/dt_without_disc.pkl")
            print("predic")
            X = dt.copy()
            print(X.shape)
            y_pred = model.predict(X)
            
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
            #Specificity Calculation
            def specificity_calc(Y,y_pred):
                cm = confusion_matrix(Y, y_pred)
                specificity = []
                for i in range(len(cm)):
                    true_negative = sum(cm[j][i] for j in range(len(cm)) if j != i)
                    false_positive = sum(cm[j][i] for j in range(len(cm)) if j != i)
                    if true_negative + false_positive == 0:
                        specificity.append(0)  # Handle the case when the denominator is zero
                    else:
                        specificity.append(true_negative / (true_negative + false_positive))
                class_counts = [sum(cm[i]) for i in range(len(cm))]
                print(class_counts)
                total_samples = sum(class_counts)
                class_weights = [count / total_samples for count in class_counts]
                specificity1 = sum(spec * weight for spec, weight in zip(specificity, class_weights))
                return specificity1
            
            #Evaluation Metrics
            report = classification_report(y_pred, y_test, output_dict=True)
            precision = report['weighted avg']['precision']
            f1_score = report['weighted avg']['f1-score']
            recall = report['weighted avg']['recall']
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            specificity1 = specificity_calc(y_test,y_pred)


            #Storing the classified files into dictionary
            file_names = list(dt.index)
            file_dict = {}
            Y_pred = list(y_pred)
            for i in range(len(file_names)):
                if file_dict.get(Y_pred[i]) == None:
                    file_dict[Y_pred[i]] = [str(file_names[i]) + '.txt']
                else:
                    file_dict[Y_pred[i]].append(str(file_names[i]) + '.txt')
            print(len(file_dict))


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
                                    "recall" : round(recall, 2),
                                    "classified_files": file_dict
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

            
            
        elif selectedalgo == "Logistic Regression" and item == "Stratified K-Fold":
            model = joblib.load("./Models/Logistic_Regression/LR-R8-STK.pkl")

            X = dt.copy()
            y_pred = model.predict(X)

            

        elif selectedalgo == "Logistic Regression" and item == "Variable Threshold":
            models = []
            model_thresholds = [0.4, 0.1, 0.8, 0.5, 0.2, 0.5, 0.1, 0.3]
            X = dt.copy()
            print(X.columns)
            classes = Y.unique()
            classes.sort()
            for i in range(len(classes)):
                print(classes[i])
                models.append(joblib.load(f'./Models/Logistic_Regression/Variable/model-' + str(i) + '.pkl'))
            y_pred = []
            probabilities = []
            for index, row in X.iterrows():
                Y_pred = []
                probas = []
                d = pd.DataFrame(row[X.columns].values.reshape(1, -1),columns = X.columns)
                d = d.sort_index(axis=1)
                for i in range(len(classes)):
                    probs = models[i].predict_proba(d)[:,1]
                    probs = probs[0]
                    probas.append(probs)
                    Y_pred.append((probs > model_thresholds[i]).astype(int))
                ind = -1
                max_proba = 0
                for j in range(len(Y_pred)):
                    if max_proba < probas[j]:
                        max_proba = probas[j]
                        ind = j
                probabilities.append(max_proba)
                y_pred.append(classes[ind])
            
        elif selectedalgo == "Logistic Regression" and item == "Gaussian":
            model = joblib.load("./Models/Logistic_Regression/LR_R8-Gaussian.pkl")

            r8_df = pd.read_csv("R8_frequency_rep.csv",index_col=0)
            r8_df = r8_df.sort_index()

            print('predict')
            labels = r8_df['Label']
            r8_df.drop('Label',axis=1,inplace=True)

            gam = 1 / r8_df.shape[1]
            similarity_matrix = rbf_kernel(r8_df, gamma=gam)

            X = similarity_matrix.copy()
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            print('prediction')
            y_pred = model.predict(X_test_scaled)
            #Specificity Calculation
            def specificity_calc(Y,y_pred):
                cm = confusion_matrix(Y, y_pred)
                specificity = []
                for i in range(len(cm)):
                    true_negative = sum(cm[j][i] for j in range(len(cm)) if j != i)
                    false_positive = sum(cm[j][i] for j in range(len(cm)) if j != i)
                    if true_negative + false_positive == 0:
                        specificity.append(0)  # Handle the case when the denominator is zero
                    else:
                        specificity.append(true_negative / (true_negative + false_positive))
                class_counts = [sum(cm[i]) for i in range(len(cm))]
                print(class_counts)
                total_samples = sum(class_counts)
                class_weights = [count / total_samples for count in class_counts]
                specificity1 = sum(spec * weight for spec, weight in zip(specificity, class_weights))
                return specificity1
            
            #Evaluation Metrics
            report = classification_report(y_pred, y_test, output_dict=True)
            precision = report['weighted avg']['precision']
            f1_score = report['weighted avg']['f1-score']
            recall = report['weighted avg']['recall']
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            specificity1 = specificity_calc(y_test,y_pred)


            #Storing the classified files into dictionary
            file_names = list(dt.index)
            file_dict = {}
            Y_pred = list(y_pred)
            for i in range(len(file_names)):
                if file_dict.get(Y_pred[i]) == None:
                    file_dict[Y_pred[i]] = [str(file_names[i]) + '.txt']
                else:
                    file_dict[Y_pred[i]].append(str(file_names[i]) + '.txt')
            print(len(file_dict))


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
                                    "recall" : round(recall, 2),
                                    "classified_files": file_dict
                                },
                            }
                        ),
                        200,
                    )

        #MLP
        elif selectedalgo == "MLP" and item == "2 Layers":
            r8_df = pd.read_csv("R8_frequency_rep.csv",index_col=0)
            r8_df = r8_df.sort_index()
            labels = r8_df['Label']
            r8_df.drop('Label',axis=1,inplace=True)

            X_train, X_test, y_train, y_test = train_test_split(r8_df, labels, test_size=0.1, random_state=42)

            X = dt.copy()
            # X = np.ascontiguousarray(X)
            scaler = StandardScaler()
            X_te = scaler.fit_transform(X_train)

            mlp = joblib.load('./Models/MLP/LR-R8-MLP-2.pkl')
            X_test_scaled = scaler.transform(X)
            y_pred = mlp.predict(X_test_scaled)

        elif selectedalgo == "MLP" and item == "3 Layers":
            r8_df = pd.read_csv("R8_frequency_rep.csv",index_col=0)
            r8_df = r8_df.sort_index()
            labels = r8_df['Label']
            r8_df.drop('Label',axis=1,inplace=True)

            X_train, X_test, y_train, y_test = train_test_split(r8_df, labels, test_size=0.1, random_state=42)

            X = dt.copy()
            # X = np.ascontiguousarray(X)
            scaler = StandardScaler()
            X_te = scaler.fit_transform(X_train)


            mlp = joblib.load('./Models/MLP/LR-R8-MLP-3.pkl')
            X_test_scaled = scaler.transform(X)
            y_pred = mlp.predict(X_test_scaled)
        
        elif selectedalgo == "MLP" and item == "4 Layers":
            r8_df = pd.read_csv("R8_frequency_rep.csv",index_col=0)
            r8_df = r8_df.sort_index()
            labels = r8_df['Label']
            r8_df.drop('Label',axis=1,inplace=True)

            X_train, X_test, y_train, y_test = train_test_split(r8_df, labels, test_size=0.1, random_state=42)

            X = dt.copy()
            # X = np.ascontiguousarray(X)
            scaler = StandardScaler()
            X_te = scaler.fit_transform(X_train)

            mlp = joblib.load('./Models/MLP/LR-R8-MLP-4.pkl')
            X_test_scaled = scaler.transform(X)
            y_pred = mlp.predict(X_test_scaled)


        #KNN
        elif selectedalgo == "KNN" and item == "Euclidean":
            knn_classifier = joblib.load('./Models/KNN/KNN-R8-Euclidean.pkl')
            
            X = dt.copy()
            x_test = np.ascontiguousarray(X)
            y_pred = knn_classifier.predict(x_test)


        elif selectedalgo == "KNN" and item == "Cosine":
            knn_classifier = joblib.load('./Models/KNN/KNN-R8-Cosine.pkl')
            
            X = dt.copy()
            x_test = np.ascontiguousarray(X)
            y_pred = knn_classifier.predict(x_test)

        else:
            return jsonify({"error": "No data found in the request body."}), 400


    #Specificity Calculation
    def specificity_calc(Y,y_pred):
        cm = confusion_matrix(Y, y_pred)
        specificity = []
        for i in range(len(cm)):
            true_negative = sum(cm[j][i] for j in range(len(cm)) if j != i)
            false_positive = sum(cm[j][i] for j in range(len(cm)) if j != i)
            if true_negative + false_positive == 0:
                specificity.append(0)  # Handle the case when the denominator is zero
            else:
                specificity.append(true_negative / (true_negative + false_positive))
        class_counts = [sum(cm[i]) for i in range(len(cm))]
        print(class_counts)
        total_samples = sum(class_counts)
        class_weights = [count / total_samples for count in class_counts]
        specificity1 = sum(spec * weight for spec, weight in zip(specificity, class_weights))
        return specificity1
    
    #Evaluation Metrics
    report = classification_report(Y, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    f1_score = report['weighted avg']['f1-score']
    recall = report['weighted avg']['recall']
    acc = accuracy_score(Y, y_pred)
    bal_acc = balanced_accuracy_score(Y, y_pred)
    specificity1 = specificity_calc(Y,y_pred)


    #Storing the classified files into dictionary
    file_names = list(dt.index)
    file_dict = {}
    Y_pred = list(y_pred)
    for i in range(len(file_names)):
        if file_dict.get(Y_pred[i]) == None:
            file_dict[Y_pred[i]] = [str(file_names[i]) + '.txt']
        else:
            file_dict[Y_pred[i]].append(str(file_names[i]) + '.txt')
    print(len(file_dict))

    ##Class Accuracies Graph
    classes = Y.unique()
    classes.sort()
    class_accuracies = {}
    for i in range(len(classes)):
        class_accuracies[classes[i]] = report[classes[i]]['recall'] * 100
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    plt.figure(figsize=(10, 6))
    plt.bar(classes, accuracies, color='skyblue')
    plt.title('Accuracy of Classes')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.savefig('class_accuracies_graph.png')



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
                            "recall" : round(recall, 2),
                            "classified_files": file_dict
                        },
                    }
                ),
                200,
            )



@app.route("/get_results", methods=["GET"])
def get_results():
    data = {
        "item": item,
        "selectedalgo": selectedalgo,
        "acc": round(acc * 100, 2),
        "bal_acc": round(bal_acc * 100, 2),
        "precision" : round(precision, 2),
        "f1_score" : round(f1_score, 2),
        "recall" : round(recall, 2),
        "specificity":round(specificity1,2),
        "classified_files":file_dict

    }
    return (
        jsonify({"message": "Data received and processed successfully.", "data": data}),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True)
