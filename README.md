# PubTrends - Dataset Clustering

A web service that analyzes and visualizes relationships between PubMed publications and their associated GEO datasets using TF-IDF clustering.

## Features

- Accepts multiple PubMed IDs (PMIDs)
- Fetches associated GEO datasets using NCBI E-utils API
- Performs TF-IDF vectorization on dataset metadata
- Clusters datasets based on content similarity
- Interactive visualization of dataset clusters
- Detailed information about each dataset and its relationships

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pubtrends-task2
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your NCBI API key:
```
NCBI_API_KEY=your_api_key_here
```

You can obtain an NCBI API key from: https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

3. Enter one or more PMIDs (one per line) in the text area and click "Analyze Datasets"

4. View the interactive visualization and detailed cluster information

## API Endpoints

- `GET /`: Home page with the web interface
- `POST /api/analyze`: Analyzes PMIDs and returns clustering results
  - Request body: `{"pmids": ["12345678", "87654321"]}`
  - Returns: Clustering results with dataset information and visualization data

## Technical Details

- Uses scikit-learn for TF-IDF vectorization and K-means clustering
- Implements cosine similarity for dataset comparison
- Visualizes clusters using Plotly.js
- Handles rate limiting and error cases for NCBI API calls
- Special handling for GDS to GSE ID mapping (e.g., PMID 30530648 returns internal ID 200116672 which maps to GSE116672)
- Fallback methods to find GSE accessions when standard API calls fail

## Error Handling

The application includes robust error handling for:
- Invalid PMIDs
- API rate limits
- Network issues
- Missing or malformed data
- Non-standard GEO ID formats and mappings

## Known Issues and Workarounds

- Some PMIDs (like 30530648) return internal GDS IDs from the eutils API instead of GSE accessions.
  The application includes special mapping to handle these cases.
- The NCBI API has rate limits. Excessive requests may be throttled. The application implements
  exponential backoff to handle this situation.
- Dataset clustering requires at least 2 valid datasets with sufficient text content.
  Single datasets will be displayed without clustering visualization.

## Contributing

Feel free to submit issues and enhancement requests!