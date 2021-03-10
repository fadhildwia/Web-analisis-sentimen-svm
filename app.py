import os
from flask import Flask, render_template, request, redirect
import csv
import pandas
from werkzeug.utils import secure_filename
from sentimen import lower, remove_punctuation, remove_stopwords, stem_text, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from yellowbrick.text import TSNEVisualizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve



app = Flask(__name__)
app.secret_key = 'uwaw'
app.run(debug=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')
    

#upload gambar
ALLOWED_EXTENSION = set(['csv'])
app.config['UPLOAD_FOLDER']='uploads'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

@app.route('/uploaddata', methods=['GET', 'POST'])
def uploaddata():
    if request.method == 'GET':
        return render_template('uploaddata.html')
    
    elif request.method == 'POST':
        file = request.files['file']
        
        if 'file' not in request.files:
            return redirect(request.url)

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            file.filename = "dataset.csv"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

            text = pandas.read_csv('uploads/dataset.csv', encoding='latin-1')
            
            return render_template('uploaddata.html',tables=[text.to_html()])


@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    return render_template ('preprocessing.html')


@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    text = pandas.read_csv('uploads/dataset.csv', encoding='latin-1')
    text.drop(['Date','Username','Location'], axis=1, inplace=True)
    
    text['Text'] = text['Text'].map(lambda x: lower(x))
    text['Text'] = text['Text'].map(lambda x: remove_punctuation(x))
    text['Text'] = text['Text'].map(lambda x: remove_stopwords(x))
    text['Text'] = text['Text'].map(lambda x: stem_text(x))

    text.to_csv('uploads/dataset_clear.csv', index = False, header = True)

    return render_template('preprocessing.html',tables=[text.to_html()])


@app.route('/tfidfpage', methods=['GET', 'POST'])
def tfidfpage():
    text = pandas.read_csv('uploads/dataset_clear.csv', encoding='latin-1')
    positif, negatif= text['Labels'].value_counts()
    total = positif + negatif
    
    return render_template ('tfidf.html', total=total, positif=positif, negatif=negatif)


def data(text):
    text['Labels'] = text['Labels'].map({'positif': 0, 'negatif': 1})
    X = text['Text']
    y = text['Labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)
    return X_train, X_test, y_train, y_test

@app.route('/tfidf', methods=['GET', 'POST'])
def tfidf():
    text = pandas.read_csv('uploads/dataset_clear.csv', encoding='latin-1')
    positif, negatif= text['Labels'].value_counts()
    total = positif + negatif

    X_train, X_test, y_train, y_test = data(text)


    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
  
    #Saving vectorizer to disk
    pickle.dump(vectorizer, open('uploads/vectorizer.model','wb'))

    return render_template ('tfidf.html', X_train=X_train, X_test=X_test, total=total, positif=positif, negatif=negatif)


@app.route('/klasifikasisvm1', methods=['GET', 'POST'])
def klasifikasisvm1():

    return render_template ('klasifikasisvm.html')


@app.route('/klasifikasisvm', methods=['GET', 'POST'])
def klasifikasisvm():
    import pickle
    # Loading model to compare the results
    vectorizer = pickle.load(open('uploads/vectorizer.model','rb'))

    text = pandas.read_csv('uploads/dataset_clear.csv', encoding='latin-1')

    X_train, X_test, y_train, y_test = data(text)

    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Process of making models Klasifikasi SVM LINEAR
    linear = SVC(kernel="linear")

    linear.fit(X_train,y_train)
    linear = linear.predict(X_test)

    # Process of making models Klasifikasi SVM RBF
    rbf = SVC(kernel="rbf")

    rbf.fit(X_train,y_train)
    rbf = rbf.predict(X_test)

    #Saving vectorizer to disk
    pickle.dump(linear, open('uploads/linear.model','wb'))
    pickle.dump(rbf, open('uploads/rbf.model','wb'))
    from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score
    # f1_score
    f1_score_linear = f1_score(y_test, linear)
    f1_score_rbf = f1_score(y_test, rbf)

    # accuracy score
    accuracy_score_linear = accuracy_score(y_test, linear)
    accuracy_score_rbf = accuracy_score(y_test, rbf)

    # precision score
    precision_score_linear = precision_score(y_test, linear)
    precision_score_rbf = precision_score(y_test, rbf)

    # recall score
    recall_score_linear = recall_score(y_test, linear)
    recall_score_rbf = recall_score(y_test, rbf)

    # confusion matrix
    tn_linear, fp_linear, fn_linear, tp_linear = confusion_matrix(y_test, linear).ravel()
    tn_rbf, fp_rbf, fn_rbf, tp_rbf = confusion_matrix(y_test, rbf).ravel()

    return render_template ('klasifikasisvm.html', f1_score_linear=f1_score_linear, accuracy_score_linear=accuracy_score_linear, precision_score_linear=precision_score_linear, recall_score_linear=recall_score_linear, 
    tn_linear=tn_linear, fp_linear=fp_linear, fn_linear=fn_linear, tp_linear=tp_linear, f1_score_rbf=f1_score_rbf, accuracy_score_rbf=accuracy_score_rbf, precision_score_rbf=precision_score_rbf, 
    recall_score_rbf=recall_score_rbf, tn_rbf=tn_rbf, fp_rbf=fp_rbf, fn_rbf=fn_rbf, tp_rbf=tp_rbf)


@app.route('/tesmodel1', methods=['GET', 'POST'])
def tesmodel1():

    return render_template ('tesmodel.html')


@app.route('/tesmodel', methods=['GET', 'POST'])
def tesmodel():
    # Loading model to compare the results
    model = pickle.load(open('uploads/model.model','rb'))
    vectorizer = pickle.load(open('uploads/vectorizer.model','rb'))

    text = request.form['text']
    original_text = request.form['text']

    # contoh kalimat 
    review_positif = "ganti biznet aja dari pada indihome"
    review_negatif = "biznet masalah dari kemarin mulu"

    hasilprepro = preprocess_data(text)
    hasiltfidf = vectorizer.transform([hasilprepro])

    # cek prediksi dari kalimat
    hasilsvm = model.predict(hasiltfidf)
    
    return render_template ('tesmodel.html', original_text=original_text, hasilprepro=hasilprepro, hasilsvm=hasilsvm)