from disaster_app import app
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

from tokenize_class.tokenize_class import tokenize_class
#from tokenize_class.tokenize_class import ColumnSelector
#app = Flask(__name__)

#def tokenize(text):
#    tokens = word_tokenize(text)
#    lemmatizer = WordNetLemmatizer()

#    clean_tokens = []
#    for tok in tokens:
#        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#        clean_tokens.append(clean_tok)

#    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("models/classifier.pkl")


# indexwebpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    correlation = df.iloc[:,4:40].corr()
    melt = pd.melt(df, id_vars=['id', 'message', 'genre'], value_vars=df.iloc[:,4:40].columns)
    grouped = melt.groupby(['genre','variable'], as_index = False)['value'].sum().sort_values(by='value', ascending= False)
    grouped_new = grouped[~grouped.variable.str.contains('related')].sort_values(by='value', ascending= False)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [      
        #Barplot ba genre
         {
            'data': [
                {
                    'x': grouped_new[grouped_new.genre == 'direct'].variable,
                    'y': grouped_new[grouped_new.genre == 'direct'].value,
                    'type': 'bar',
                    'color': 'rrgba(255,153,51,0.6)',
                    'name': 'direct'
                    
                },
                {
                    'x': grouped_new[grouped_new.genre == 'news'].variable,
                    'y': grouped_new[grouped_new.genre == 'news'].value,
                    'color': 'rgba(55,128,191,0.6)',
                    'type': 'bar',
                    'name': 'news'
                
                },
                {
                    'x': grouped_new[grouped_new.genre == 'social'].variable,
                    'y': grouped_new[grouped_new.genre == 'social'].value,
                    'color': 'rgba(55,128,191,0.6)',
                    'type': 'bar',
                    'name': 'social'
                
                }
            ],

            'layout': {
                'barmode': 'stack',
                'title': 'Messages by cluster and genre',
                'yaxis': {
                    'title': "Number of messages"
                },
                'xaxis': {
                    'title': "clusters"
                },
            }
        },
       
        #heatmap correlation by cluster
         {
            'data': [
                {
                    'z': correlation.values,
                    'x': correlation.columns.tolist(),
                    'y': correlation.columns.tolist(),
                    'type': 'heatmap',
                    'colorscale': 'YlOrRd'
                }
            ],

            'layout': {
                'title': 'Correlation of clusters',
                'yaxis': {
                    'title': "clusters"
                },
                'xaxis': {
                    'title': "clusters"
                },
                #'width': 800,
                'height': 800
            }
        }

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


#def main():
#    #app.run(host='0.0.0.0', port=3001, debug=True)
#    pass
#
#
#if __name__ == '__main__':
#    main()