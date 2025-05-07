#!/bin/bash

echo "🚀 Starting setup for Research Paper Q&A Assistant..."

# Clone the repository
if [ ! -d "research_paper_RAG_chain" ]; then
    git clone https://github.com/ajay-del-bot/research_paper_RAG_chain.git
fi
cd research_paper_RAG_chain || exit 1

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating a template .env..."
    cat <<EOL > .env
PINECONE_API_KEY=YOUR_PINECONE_API_KEY
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
INDEX_NAME='test-db'
EOL
    echo "✅ Please update the .env file with your actual API keys before proceeding."
    exit 1
fi

# Run the Flask server
echo "✅ Environment setup complete. Running the server..."
python3 src/server.py