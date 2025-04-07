from flask import Flask, request, render_template, jsonify
import requests
import os
from lxml import etree
from time import sleep
import logging
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import json
import re

load_dotenv()

app = Flask(__name__)
API_KEY = os.getenv('NCBI_API_KEY')
HEADERS = {'User-Agent': 'PubTrends/1.0 (muhammadjonprogrammer12@gmail.com)'}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_pmids():
    try:
        data = request.get_json()
        pmids = data.get('pmids', [])
        
        if not pmids:
            return jsonify({'error': 'No PMIDs provided'}), 400
            
        # Fetch GEO data for all PMIDs
        all_datasets = []
        pmid_to_datasets = {}
        
        for pmid in pmids:
            geo_ids = get_geo_ids(pmid)
            datasets = []
            
            for gse_id in geo_ids:
                metadata = get_geo_metadata(gse_id)
                if metadata:
                    dataset = {
                        'gse_id': gse_id,
                        'pmid': pmid,
                        **metadata
                    }
                    datasets.append(dataset)
                    all_datasets.append(dataset)
            
            pmid_to_datasets[pmid] = datasets
        
        if not all_datasets:
            return jsonify({'error': 'No GEO datasets found for the provided PMIDs'}), 404
            
        # Prepare text for TF-IDF
        texts = []
        for dataset in all_datasets:
            # Combine text fields with proper preprocessing
            text_fields = [
                dataset['title'],
                dataset['type'],
                dataset['summary'],
                dataset['organism'],
                dataset['design']
            ]
            # Clean and combine text fields
            combined_text = ' '.join([clean_text(field) for field in text_fields if field])
            if not combined_text.strip():
                continue
            texts.append(combined_text)
            
        if not texts:
            return jsonify({'error': 'No valid text data found in datasets'}), 400
            
        # Create TF-IDF vectors with minimal preprocessing
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            min_df=1,  # Include terms that appear in at least 1 document
            max_df=0.95,  # Exclude terms that appear in more than 95% of documents
            token_pattern=r'(?u)\b\w+\b'  # Match any word character
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError as e:
            logger.error(f"TF-IDF vectorization error: {str(e)}")
            return jsonify({'error': 'Failed to process text data'}), 500
            
        # Perform dimensionality reduction for visualization
        if tfidf_matrix.shape[1] > 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(tfidf_matrix.toarray())
        else:
            coords = tfidf_matrix.toarray()
            
        # Perform clustering
        n_clusters = min(5, len(texts))  # Use 5 clusters or less if fewer datasets
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Prepare response data
        response_data = {
            'datasets': all_datasets,
            'clusters': clusters.tolist(),
            'coordinates': coords.tolist(),  # Use PCA coordinates instead of similarity matrix
            'pmid_to_datasets': pmid_to_datasets
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def clean_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def get_geo_ids(pmid):
    """Fetch GEO IDs with rate limit handling"""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&db=gds&linkname=pubmed_gds&id={pmid}&api_key={API_KEY}"
    
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            
            # Parse XML safely
            parser = etree.XMLParser(recover=True)
            root = etree.fromstring(response.content, parser=parser)
            return root.xpath("//Link/Id/text()")
            
        except Exception as e:
            if attempt == 2:
                logger.error(f"Failed to get GEO IDs for {pmid}: {str(e)}")
                return []
            sleep(2 ** attempt)  # Exponential backoff

def get_geo_metadata(gse_id):
    """Robust GEO metadata fetcher"""
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}&form=xml&api_key={API_KEY}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        # Handle malformed XML
        parser = etree.XMLParser(recover=True, remove_blank_text=True)
        root = etree.fromstring(response.content, parser=parser)
        
        return {
            'title': safe_extract(root, ".//Title"),
            'summary': safe_extract(root, ".//Summary"),
            'type': safe_extract(root, ".//Type"),
            'organism': safe_extract(root, ".//Organism"),
            'design': safe_extract(root, ".//Overall-Design")
        }
    except Exception as e:
        logger.error(f"Error processing {gse_id}: {str(e)}")
        return None

def safe_extract(root, path):
    """Safe XML element extraction"""
    element = root.find(path)
    return element.text.strip() if element is not None and element.text else ""

if __name__ == '__main__':
    app.run(debug=True)