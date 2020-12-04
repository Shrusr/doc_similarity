import transformers
from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
import faiss
import numpy as np
from numpy import load
from flask import Flask,request,jsonify
from flask_cors import CORS, cross_origin



tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
data = pd.read_csv('ticket_data.csv')
documents = data['Summary'].to_list()
doc_links =  data['Issue key'].to_list()
sent_embeddings = load('sent_embeddings_tickets.npy')
index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
index.add_with_ids(sent_embeddings,np.array(range(0, len(sent_embeddings))).astype('int64'))


def encode(document: str) -> torch.Tensor:
    tokens = tokenizer(document, return_tensors='pt')
    vector = model(**tokens)[0].detach().squeeze()
    return torch.mean(vector, dim=0)

def get_similar_docs(query: str, k=2):
    encoded_query = encode(query).unsqueeze(dim=0).numpy()
    top_k = index.search(encoded_query, k)
    scores = top_k[0][0]
    results = [documents[_id] for _id in top_k[1][0]]   
    links = [doc_links[_id] for _id in top_k[1][0]]
    return [{"Document":result,"score":float(score),"Link":link} for result, score, link in zip(results,scores,links)]

def get_distinace(q1,q2):
    q1_encode = np.array(encode(q1)).astype(float)
    q2_encode = np.array(encode(q2)).astype(float)
    dist = np.linalg.norm(q1_encode - q2_encode)
    return dist    

print(get_similar_docs("jira licence",k=3))    

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/query',methods=["POST"])
@cross_origin()
def hello_world():
    query = request.args.get('query')
    k = request.args.get('k')
    print(query)
    result = get_similar_docs(query,int(k))
    return jsonify(result)

@app.route('/dist',methods=["POST"])
@cross_origin()
def dist_call():
    request_json = request.get_json()
    df = pd.DataFrame.from_records(request_json)
    q1 = df['q1'][0]
    q2 = df['q2'][0]
    result1 = get_distinace(q1,q2)
    return jsonify("distance" + " " + str(result1))    
    
app.run(host="0.0.0.0",port=80)