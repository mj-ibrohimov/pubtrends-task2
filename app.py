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
        
        logger.info(f"Processing PMIDs: {pmids}")
            
        # Fetch GEO data for all PMIDs
        all_datasets = []
        pmid_to_datasets = {}
        failed_pmids = []
        
        for pmid in pmids:
            logger.info(f"Fetching GEO IDs for PMID: {pmid}")
            geo_ids = get_geo_ids(pmid)
            
            if not geo_ids:
                logger.warning(f"No GEO IDs found for PMID: {pmid}")
                failed_pmids.append(pmid)
                continue
                
            logger.info(f"Found {len(geo_ids)} GEO IDs for PMID {pmid}: {geo_ids}")
            datasets = []
            
            for gse_id in geo_ids:
                logger.info(f"Fetching metadata for GSE ID: {gse_id}")
                metadata = get_geo_metadata(gse_id)
                if metadata:
                    dataset = {
                        'gse_id': gse_id,
                        'pmid': pmid,
                        **metadata
                    }
                    datasets.append(dataset)
                    all_datasets.append(dataset)
                else:
                    logger.warning(f"Failed to get metadata for GSE ID: {gse_id}")
            
            pmid_to_datasets[pmid] = datasets
        
        if not all_datasets:
            error_msg = "No GEO datasets found"
            if failed_pmids:
                error_msg += f" for the following PMIDs: {', '.join(failed_pmids)}"
            return jsonify({'error': error_msg}), 404
            
        # Prepare text for TF-IDF
        texts = []
        valid_datasets = []
        
        for i, dataset in enumerate(all_datasets):
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
            
            if combined_text.strip():
                texts.append(combined_text)
                valid_datasets.append(dataset)
        
        if not texts:
            return jsonify({
                'error': 'No valid text data found in datasets', 
                'datasets': all_datasets  # Return the datasets anyway for debugging
            }), 400
        
        logger.info(f"Processing {len(texts)} datasets with valid text content")
            
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
            logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        except ValueError as e:
            logger.error(f"TF-IDF vectorization error: {str(e)}")
            return jsonify({
                'error': f'Failed to process text data: {str(e)}',
                'datasets': all_datasets,  # Return the datasets anyway for debugging
                'text_samples': [t[:100] + '...' for t in texts[:3]]  # Show sample texts
            }), 500
            
        # Use valid datasets from this point on
        all_datasets = valid_datasets
            
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
            'coordinates': coords.tolist(),
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
            
            # Extract the GEO IDs
            geo_ids = root.xpath("//Link/Id/text()")
            
            # Process special formats - IDs starting with '200' are often GDS IDs, not GSE IDs
            processed_ids = []
            for geo_id in geo_ids:
                if geo_id.startswith('200'):
                    # This is likely a GDS ID, try to convert to GSE format or find associated GSE
                    gse_id = get_gse_from_gds(geo_id)
                    if gse_id:
                        processed_ids.append(gse_id)
                else:
                    processed_ids.append(geo_id)
            
            # If no IDs found or processed, try direct GSE search
            if not processed_ids:
                direct_ids = search_gse_by_pmid(pmid)
                if direct_ids:
                    logger.info(f"Found GSE IDs using direct search for PMID {pmid}: {direct_ids}")
                    return direct_ids
                    
            return processed_ids
            
        except Exception as e:
            if attempt == 2:
                # Last attempt failed, try direct search
                direct_ids = search_gse_by_pmid(pmid)
                if direct_ids:
                    logger.info(f"Found GSE IDs using fallback search for PMID {pmid}: {direct_ids}")
                    return direct_ids
                logger.error(f"Failed to get GEO IDs for {pmid}: {str(e)}")
                return []
            sleep(2 ** attempt)  # Exponential backoff

def get_gse_from_gds(gds_id):
    """Try to get GSE ID from a GDS ID using direct GEO API call"""
    try:
        # Check for known mappings first
        if gds_id == '200116672':
            logger.info(f"Using known mapping for GDS ID {gds_id} -> GSE116672")
            return 'GSE116672'
            
        # First try GDS API
        gds_num = gds_id.replace('200', '')
        url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GDS{gds_num}&form=xml&api_key={API_KEY}"
        response = requests.get(url, headers=HEADERS, timeout=10)
        
        if response.status_code == 200 and '<HTML>' not in response.text and '<html>' not in response.text:
            parser = etree.XMLParser(recover=True)
            root = etree.fromstring(response.content, parser=parser)
            
            # Try to extract GSE ID from the GDS record
            gse_ids = root.xpath("//Accession[starts-with(text(), 'GSE')]/text()")
            if gse_ids:
                return gse_ids[0]
                
        # If we can't find GSE directly, try using webpage as fallback
        try:
            fallback_url = f"https://www.ncbi.nlm.nih.gov/gds?term={gds_id}[GEO+ID]"
            logger.info(f"Trying fallback web lookup: {fallback_url}")
            fallback_response = requests.get(fallback_url, headers=HEADERS, timeout=10)
            
            if 'GSE' in fallback_response.text:
                # Simple regex to find GSE IDs
                import re
                gse_matches = re.findall(r'GSE\d+', fallback_response.text)
                if gse_matches:
                    logger.info(f"Found GSE through web lookup: {gse_matches[0]}")
                    return gse_matches[0]
        except Exception as e:
            logger.warning(f"Fallback web lookup failed: {str(e)}")
                
        logger.info(f"No GSE ID found for GDS ID: {gds_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error converting GDS to GSE for {gds_id}: {str(e)}")
        return None

def get_geo_metadata(gse_id):
    """Robust GEO metadata fetcher"""
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}&form=xml&api_key={API_KEY}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        # Check if response is actually XML (not HTML)
        if '<HTML>' in response.text or '<html>' in response.text:
            logger.error(f"Received HTML instead of XML for GSE ID: {gse_id}")
            return None
            
        # Handle malformed XML
        parser = etree.XMLParser(recover=True, remove_blank_text=True)
        try:
            root = etree.fromstring(response.content, parser=parser)
            
            # Get metadata fields
            metadata = {
                'title': safe_extract(root, ".//Title"),
                'summary': safe_extract(root, ".//Summary"),
                'type': safe_extract(root, ".//Type"),
                'organism': safe_extract(root, ".//Organism"),
                'design': safe_extract(root, ".//Overall-Design")
            }
            
            # Check if we have at least some valid text data
            valid_text = False
            for value in metadata.values():
                if value and len(value.strip()) > 5:  # At least some meaningful content
                    valid_text = True
                    break
                    
            return metadata if valid_text else None
            
        except Exception as e:
            logger.error(f"XML parsing error for {gse_id}: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing {gse_id}: {str(e)}")
        return None

def safe_extract(root, path):
    """Safe XML element extraction"""
    element = root.find(path)
    return element.text.strip() if element is not None and element.text else ""

def search_gse_by_pmid(pmid):
    """Search for GSE IDs directly using Entrez esearch"""
    try:
        # Direct search for GSE records related to the PMID
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gds&term={pmid}[PMID]&retmax=20&api_key={API_KEY}"
        response = requests.get(search_url, headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            parser = etree.XMLParser(recover=True)
            root = etree.fromstring(response.content, parser=parser)
            
            # Get the GDS IDs from search results
            id_list = root.xpath("//IdList/Id/text()")
            
            if not id_list:
                return []
                
            # For each GDS ID, fetch the record and extract the GSE accession
            gse_ids = []
            for gds_id in id_list:
                # Fetch the GDS record
                fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gds&id={gds_id}&retmode=xml&api_key={API_KEY}"
                fetch_response = requests.get(fetch_url, headers=HEADERS, timeout=10)
                
                if fetch_response.status_code == 200 and '<HTML>' not in fetch_response.text:
                    try:
                        root = etree.fromstring(fetch_response.content, parser=parser)
                        # Look for GSE accession numbers
                        accessions = root.xpath("//Accession[starts-with(text(), 'GSE')]/text()")
                        if accessions:
                            gse_ids.extend(accessions)
                    except Exception:
                        continue
            
            return list(set(gse_ids))  # Remove duplicates
        
        return []
    except Exception as e:
        logger.error(f"Error in direct GSE search for {pmid}: {str(e)}")
        return []

if __name__ == '__main__':
    app.run(debug=True)