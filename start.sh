#!/bin/bash

set -e

function generate_random_string() { 
    length=32
    random_string=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c ${length})
    echo "${random_string}"
}

if [ -z "$API_SECRET" ]; then
    API_SECRET="$(generate_random_string)"
    echo "API_SECRET set to ${API_SECRET}"
    export API_SECRET
fi

if [[ $PUBLIC_KEY ]]
then
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    cd ~/.ssh
    echo "$PUBLIC_KEY" >> authorized_keys
    chmod 700 -R ~/.ssh
    cd /
    service ssh start
else
    echo "No public key provided, skipping ssh setup"
fi

echo "Starting ollama"
ollama serve &

while ! curl "$OLLAMA_URL"; do 
    sleep 1 
done

echo "Pulling $OLLAMA_MODEL_NAME from ollama library"
ollama pull "$OLLAMA_MODEL_NAME"

cd /workspace

echo "Starting api"
uvicorn wikidatachat:app --reload --host 0.0.0.0 --port 8000 &

echo "Ready"
sleep infinity
