import os
import yaml
from flask import Flask, render_template, request

import firebase_admin
from firebase_admin import firestore

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine import MatchingEngineIndexEndpoint

# Instantiating the Firebase client
firebase_app = firebase_admin.initialize_app()
db = firestore.client()

# Instantiate an embedding model here
embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

# Instantiate a Generative AI model here
gen_model = GenerativeModel("gemini-pro")

# Helper function that reads from the config file. 
def get_config_value(config, section, key, default=None):
    """
    Retrieve a configuration value from a section with an optional default value.
    """
    try:
        return config[section][key]
    except:
        return default

# Open the config file (config.yaml)
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Read application variables from the config fle
TITLE = get_config_value(config, 'app', 'title', 'Ask Google')
SUBTITLE = get_config_value(config, 'app', 'subtitle', 'Your friendly Bot')
CONTEXT = get_config_value(config, 'palm', 'context',
                           'You are a bot who can answer all sorts of questions')
BOTNAME = get_config_value(config, 'palm', 'botname', 'Google')
TEMPERATURE = get_config_value(config, 'palm', 'temperature', 0.8)
MAX_OUTPUT_TOKENS = get_config_value(config, 'palm', 'max_output_tokens', 256)
TOP_P = get_config_value(config, 'palm', 'top_p', 0.8)
TOP_K = get_config_value(config, 'palm', 'top_k', 40)


app = Flask(__name__)

# The Home page route
@app.route("/", methods=['POST', 'GET'])
def main():

    # The user clicked on a link to the Home page
    # They haven't yet submitted the form
    if request.method == 'GET':
        question = ""
        answer = "Hi, I'm FreshBot. How may I be of assistance to you?"

    # The user asked a question and submitted the form
    # The request.method would equal 'POST'
    else: 
        question = request.form['input']

        # Get the data to answer the question that 
        # most likely matches the question based on the embeddings
        data = search_vector_database(question)

        # Ask Gemini to answer the question using the data 
        # from the database
        answer = ask_gemini(question, data)
        
    # Display the home page with the required variables set
    model = {"title": TITLE, "subtitle": SUBTITLE,
             "botname": BOTNAME, "message": answer, "input": question}
    return render_template('index.html', model=model)


def search_vector_database(question):

    # 1. Convert the question into an embedding
    embeddings = embedding_model.get_embeddings([TextEmbeddingInput(text=question)])
    embedding_vector = embeddings[0].values
    # 2. Search the Vector database for the 5 closest embeddings to the user's question
    index_endpoint_name = 'projects/679836286533/locations/us-central1/indexEndpoints/6638714869069643776'
    my_index_endpoint = MatchingEngineIndexEndpoint(index_endpoint_name=index_endpoint_name)
    
    response = my_index_endpoint.find_neighbors(
	deployed_index_id = "assessment_index_deployed",
        queries=[embedding_vector],
        num_neighbors=5
    )
    # 3. Get the IDs for the five embeddings that are returned
    neighbor_ids = [neighbor.id for neighbors in response for neighbor in neighbors]

    # 4. Get the five documents from Firestore that match the IDs
    documents = []
    for doc_id in neighbor_ids:
        doc_ref = db.collection('pdf_pages').document(doc_id)
        doc = doc_ref.get()
        if doc.exists:
            documents.append(doc.to_dict().get('content', ''))
    # 5. Concatenate the documents into a single string and return it
    data = "".join(documents)
    return data


def ask_gemini(question, data):
    # You will need to change the code below to ask Gemni to
    # answer the user's question based on the data retrieved
    # from their search
    prompt = f"""
    Instructions: Use the data provided to answer the question. Only answer questions that are related to the data.
    
    Data: {data}
    
    Question: {question}
    
    Answer:
    """

    responses = gen_model.generate_content(prompt,generation_config={"max_output_tokens": 2048,"temperature": 0.9,"top_p": 1}, stream=True,)
    
    generated_responses = []

    # Iterate over the generator to collect all responses
    for response in responses:
        generated_responses.append(response.text.strip())  # Assuming response has a text attribute

    # Return the first generated response (or concatenate all as per your requirement)
    return "\n".join(generated_responses)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
