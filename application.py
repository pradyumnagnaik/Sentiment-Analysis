from flask import Flask, render_template, request
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import accuracy_score,classification_report, ConfusionMatrixDisplay


# Ignore FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)


app = Flask(__name__, template_folder='templates')



# Define the 'd' DataFrame using the 'train.csv' file
d = pd.read_csv('train.csv', encoding='latin1')

# Define the 'f' DataFrame using the 'test.csv' file
f = pd.read_csv('test.csv', encoding='latin1')

df = pd.concat([d, f])  # Ensure that you have 'd' and 'f' DataFrames defined


df.dropna(inplace=True)

df['sentiment'].value_counts(normalize=True).plot(kind='bar');

# Convert sentiment column to categorical variable
df['sentiment'] = df['sentiment'].astype('category').cat.codes
df['sentiment'].value_counts(normalize=True).plot(kind='bar');

# Convert Time of Tweet column to categorical variable
df['Time of Tweet'] = df['Time of Tweet'].astype('category').cat.codes
# Convert Country column to categorical variable
df['Country'] = df['Country'].astype('category').cat.codes
# convert Age of User to integer 
df['Age of User']=df['Age of User'].replace({'0-20':18,'21-30':25,'31-45':38,'46-60':53,'60-70':65,'70-100':80})

def wp(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

df.drop(columns=['textID','Time of Tweet', 'Age of User', 'Country', 'Population -2020', 'Land Area (Km²)', 'Density (P/Km²)'])

X=df['selected_text']
y= df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


vectorization = TfidfVectorizer()
XV_train = vectorization.fit_transform(X_train)
XV_test = vectorization.transform(X_test)

lr = LogisticRegression(n_jobs=-1)
lr.fit(XV_train,y_train)

pred_lr=lr.predict(XV_test)

# get accuracy score
score_lr = accuracy_score(y_test, pred_lr)
score_lr

dt = DecisionTreeClassifier()
dt.fit(XV_train, y_train)

pred_dt = dt.predict(XV_test)

score_dt = dt.score(XV_test, y_test)
score_dt

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        processed_text = wp(text)
        processed_text_vectorized = vectorization.transform([processed_text])

        pred_lr = lr.predict(processed_text_vectorized)
        pred_dt = dt.predict(processed_text_vectorized)

        sentiment_lr = output_lable(pred_lr[0])
        sentiment_dt = output_lable(pred_dt[0])

        return render_template('index.html', sentiment_lr=sentiment_lr, sentiment_dt=sentiment_dt, text=text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)