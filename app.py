#!/usr/bin/env python
# coding: utf-8

# In[7]:


from flask import Flask,request,jsonify
from flask_cors import CORS
import recommendation


# In[14]:


app = Flask(__name__)
CORS(app) 
        
@app.route('/movie', methods=['GET'])
def recommend_movies():
    res = recommendation.results(request.args.get('title'))
    return jsonify(res)

if __name__=='__main_':
    app.run(port = 5000, debug = True)


# In[ ]:




