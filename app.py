import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the Dash app
app = dash.Dash(__name__)

# Step 1: Prepare the Dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data
labels = newsgroups.target

# Step 2: Build the Term-Document Matrix using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
term_doc_matrix = vectorizer.fit_transform(documents)

# Step 3: Apply SVD for Dimensionality Reduction (LSA)
n_components = 100  # Number of components to keep
svd_model = TruncatedSVD(n_components=n_components)
lsa_matrix = svd_model.fit_transform(term_doc_matrix)

# Step 4: Set up the User Interface (Web Page Layout)
app.layout = html.Div([
    html.H1("Latent Semantic Analysis (LSA) Search Engine", style={'textAlign': 'center', 'color': 'blue'}),

    # Input field for user query
    html.Div([
        dcc.Input(id='query-input', type='text', placeholder='Enter your query...', style={'width': '400px', 'padding': '5px'}),
        html.Button('Search', id='search-button', n_clicks=0, style={'padding': '5px', 'marginLeft': '10px'})
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    # Section to display search results (initially blank)
    html.H2("Results", id='results-heading', style={'textAlign': 'left', 'display': 'none'}),
    html.Div(id='results-output', style={'textAlign': 'left', 'marginBottom': '20px'}),

    # Bar chart to display cosine similarity of top documents (initially blank)
    dcc.Graph(id='similarity-graph', style={'display': 'none'})
], style={'padding': '20px'})  # Added padding for better spacing

# Step 5: Handle User Queries and Perform Search
@app.callback(
    [Output('results-output', 'children'),
     Output('results-heading', 'style'),
     Output('similarity-graph', 'style'),
     Output('similarity-graph', 'figure')],
    [Input('search-button', 'n_clicks')],
    [State('query-input', 'value')]
)
def perform_search(n_clicks, query):
    if query is None or query.strip() == "":
        return "Please enter a query to search.", {'display': 'none'}, {'display': 'none'}, {}

    # Vectorize the query using the same vectorizer
    query_vector = vectorizer.transform([query])

    # Transform the query using the same SVD model (LSA)
    query_lsa_vector = svd_model.transform(query_vector)

    # Compute cosine similarities between the query and the documents
    cosine_similarities = cosine_similarity(query_lsa_vector, lsa_matrix).flatten()

    # Retrieve the top 5 most similar documents
    top_5_indices = cosine_similarities.argsort()[-5:][::-1]
    top_5_docs = [documents[i] for i in top_5_indices]
    top_5_similarities = cosine_similarities[top_5_indices]

    # Display the top 5 results (no highlighting, just plain text)
    result_output = []
    for i in range(5):
        result_output.append(
            html.Div([
                html.H4(f"Document {top_5_indices[i]}"),
                html.P(top_5_docs[i][:500]),  # First 500 characters of the document
                html.P(f"Similarity: {top_5_similarities[i]:.6f}"),
            ], style={'border': '1px solid #ddd', 'padding': '10px', 'margin': '10px 0', 'textAlign': 'left'})  # Left-aligned results
        )

    # Create a bar chart to display cosine similarities of the top 5 documents with scores on top
    bar_chart = go.Figure([go.Bar(x=[f"Doc {i+1}" for i in range(5)], y=top_5_similarities, text=[f"{sim:.6f}" for sim in top_5_similarities], textposition='auto')])
    bar_chart.update_layout(title="Cosine Similarity of Top Documents",
                            xaxis_title="Documents", yaxis_title="Cosine Similarity")

    return result_output, {'display': 'block'}, {'display': 'block'}, bar_chart

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=3000)
