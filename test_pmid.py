#!/usr/bin/env python3
"""
Test script to debug PMID to GEO ID mapping and metadata retrieval
Usage: python test_pmid.py <pmid>
"""

import sys
import requests
import os
from lxml import etree
from dotenv import load_dotenv
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

API_KEY = os.getenv('NCBI_API_KEY', '')
HEADERS = {'User-Agent': 'PubTrends/1.0 (muhammadjonprogrammer12@gmail.com)'}

def get_geo_ids(pmid):
    """Fetch GEO IDs using eutils API"""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&db=gds&linkname=pubmed_gds&id={pmid}&api_key={API_KEY}"
    logger.info(f"Fetching GEO IDs from: {url}")
    
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    
    # Save raw response for debugging
    with open(f"pmid_{pmid}_elink_response.xml", "w") as f:
        f.write(response.text)
    
    # Parse XML
    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(response.content, parser=parser)
    
    # Get all IDs from Link elements
    geo_ids = root.xpath("//Link/Id/text()")
    logger.info(f"Found {len(geo_ids)} GEO IDs: {geo_ids}")
    
    # Special handling for IDs starting with 200
    processed_ids = []
    for geo_id in geo_ids:
        if geo_id.startswith('200'):
            # This is likely a GDS ID
            logger.info(f"ID {geo_id} appears to be a GDS ID, attempting to find GSE")
            try:
                # Try to get the GSE from the GDS ID
                gds_num = geo_id.replace('200', '')
                gds_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GDS{gds_num}&form=xml&api_key={API_KEY}"
                logger.info(f"Fetching GDS info from: {gds_url}")
                
                gds_response = requests.get(gds_url, headers=HEADERS, timeout=10)
                
                # Save GDS response for debugging
                with open(f"gds_{gds_num}_response.xml", "w") as f:
                    f.write(gds_response.text)
                
                if '<HTML>' not in gds_response.text:
                    gds_root = etree.fromstring(gds_response.content, parser=parser)
                    gse_ids = gds_root.xpath("//Accession[starts-with(text(), 'GSE')]/text()")
                    if gse_ids:
                        logger.info(f"Found GSE ID from GDS{gds_num}: {gse_ids[0]}")
                        processed_ids.append(gse_ids[0])
                    else:
                        logger.warning(f"No GSE ID found in GDS{gds_num}")
                else:
                    logger.warning(f"Received HTML instead of XML for GDS{gds_num}")
            except Exception as e:
                logger.error(f"Error processing GDS ID {geo_id}: {str(e)}")
        else:
            processed_ids.append(geo_id)
    
    return processed_ids

def direct_gse_search(pmid):
    """Direct search for GSE records using esearch"""
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gds&term={pmid}[PMID]&retmax=20&api_key={API_KEY}"
    logger.info(f"Direct search for GSE with: {search_url}")
    
    response = requests.get(search_url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    
    # Save response for debugging
    with open(f"pmid_{pmid}_esearch_response.xml", "w") as f:
        f.write(response.text)
    
    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(response.content, parser=parser)
    
    # Get ID list
    id_list = root.xpath("//IdList/Id/text()")
    logger.info(f"Found {len(id_list)} GDS IDs in direct search: {id_list}")
    
    gse_ids = []
    for gds_id in id_list:
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gds&id={gds_id}&retmode=xml&api_key={API_KEY}"
        logger.info(f"Fetching details for ID {gds_id} from: {fetch_url}")
        
        fetch_response = requests.get(fetch_url, headers=HEADERS, timeout=10)
        
        # Save each efetch response for debugging
        with open(f"gds_id_{gds_id}_efetch_response.xml", "w") as f:
            f.write(fetch_response.text)
        
        if '<HTML>' not in fetch_response.text:
            try:
                root = etree.fromstring(fetch_response.content, parser=parser)
                # Get GSE IDs
                accessions = root.xpath("//Accession[starts-with(text(), 'GSE')]/text()")
                if accessions:
                    logger.info(f"Found GSE IDs for GDS {gds_id}: {accessions}")
                    gse_ids.extend(accessions)
                else:
                    logger.warning(f"No GSE accessions found for GDS ID {gds_id}")
            except Exception as e:
                logger.error(f"Error parsing XML for GDS ID {gds_id}: {str(e)}")
    
    return list(set(gse_ids))  # Remove duplicates

def get_geo_metadata(gse_id):
    """Get metadata for a GSE ID"""
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}&form=xml&api_key={API_KEY}"
    logger.info(f"Fetching metadata for GSE ID {gse_id} from: {url}")
    
    response = requests.get(url, headers=HEADERS, timeout=15)
    
    # Save response for debugging
    with open(f"gse_{gse_id}_metadata_response.xml", "w") as f:
        f.write(response.text)
    
    if '<HTML>' in response.text or '<html>' in response.text:
        logger.error(f"Received HTML instead of XML for GSE ID: {gse_id}")
        return None
    
    try:
        parser = etree.XMLParser(recover=True, remove_blank_text=True)
        root = etree.fromstring(response.content, parser=parser)
        
        # Extract metadata fields
        metadata = {
            'title': safe_extract(root, ".//Title"),
            'summary': safe_extract(root, ".//Summary"),
            'type': safe_extract(root, ".//Type"),
            'organism': safe_extract(root, ".//Organism"),
            'design': safe_extract(root, ".//Overall-Design")
        }
        
        # Log the extracted fields
        for field, value in metadata.items():
            logger.info(f"Field: {field}, Value: {value[:100]}..." if len(value) > 100 else f"Field: {field}, Value: {value}")
        
        return metadata
    except Exception as e:
        logger.error(f"Error parsing XML for GSE ID {gse_id}: {str(e)}")
        return None

def safe_extract(root, path):
    """Safely extract text from XML element"""
    element = root.find(path)
    if element is not None and element.text:
        return element.text.strip()
    return ""

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pmid.py <pmid>")
        sys.exit(1)
    
    pmid = sys.argv[1]
    logger.info(f"Testing PMID: {pmid}")
    
    # Get GEO IDs using eutils
    geo_ids = get_geo_ids(pmid)
    logger.info(f"Final GEO IDs from elink: {geo_ids}")
    
    # Try direct search if no GEO IDs found
    if not geo_ids:
        logger.info("No GEO IDs found through elink. Trying direct search...")
        geo_ids = direct_gse_search(pmid)
        logger.info(f"Final GEO IDs from direct search: {geo_ids}")
    
    # Get metadata for each GEO ID
    metadata_results = []
    for gse_id in geo_ids:
        metadata = get_geo_metadata(gse_id)
        if metadata:
            metadata_results.append({
                'gse_id': gse_id,
                'pmid': pmid,
                **metadata
            })
    
    # Save final results to JSON
    with open(f"pmid_{pmid}_results.json", "w") as f:
        json.dump(metadata_results, f, indent=2)
    
    logger.info(f"Found {len(metadata_results)} datasets with metadata for PMID {pmid}")
    logger.info(f"Results saved to pmid_{pmid}_results.json")

if __name__ == "__main__":
    main() 