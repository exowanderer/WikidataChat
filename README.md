# WikidataChat
Retrieval Augmented Generation (RAG) with for answering question with the Wikidata REST API

## Introduction
WikidataQAChat is an innovative tool designed to leverage the comprehensive knowledge base of Wikidata, transforming it into a user-friendly question-and-answer chat interface. It aims to provide the general public with validated and cited knowledge directly from Wikidata, reducing the chances of misinformation or "hallucinations" often associated with large language models (LLMs).

## Features
WikidataQAChat boasts a unique textification pipeline with the following capabilities:
- **Search and Download**: Utilizes Google's Serapi search pipeline and Wikidata's REST API to fetch relevant JSON data.
- **Textification**: Converts Wikidata statements into string statements, preparing them for processing.
- **RAG Pipeline**: Merges Wikidata string statements with user-provided questions to generate informed and accurate answers through an LLM.
- **User Interface**: Offers a friendly UI that not only presents answers but also provides linked lists of Wikidata and Wikipedia URLs for further exploration.

![Wikidata and the Meaning of Life](https://github.com/exowanderer/WikidataChat/blob/main/images/wikidatachat_meaning_of_life_example_mar25_2024.png)

## Getting Started

### Prerequisites
- Docker installed on your system or an active Python environment.

### Installation
Deploy WikidataQAChat using Docker with the following commands:
```bash
DOCKER_BUILDKIT=1 docker build . -t wdchat

docker run  \
  --env HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
  --env SERAPI_API_KEY=$SERAPI_API_KEY \
  --volume "$(pwd)/wikidatachat":/workspace/wikidatachat \
  --volume wdchat_cache:/root/.cache \
  --publish 8000:8000 \
  --rm \
  --interactive \
  --tty \
  --name wdchat \
  wdchat
```

This will deploy the UI to `localhost:8000`, allowing local access to the WikidataQAChat interface.

## Usage
After installation, access the WikidataQAChat through your web browser by navigating to `localhost:8000`. If deployed on an internet-accessible server, the interface can be accessed from the respective web address, providing a seamless experience for asking questions and receiving answers.

The UI and Haystack functionality were developed with colleagues from Wikimedia Deutschland
- [Haystack Pipeline: rti](https://github.com/rti/gbnc/)
- [Vue3 UI: andrewtavis](https://github.com/andrewtavis/gbnc)

## Contributing
We welcome contributions from the community. Whether it's features, bug fixes, or documentation, here's how you can help:
1. Fork the repository and create a branch for your contribution. Use descriptive names like `feature/streamlined_rag_pipeline` for features or `bug/localhost_is_blank` for bug fixes.
2. Submit pull requests with your changes. Make sure your contributions are narrowly defined and clearly described.
3. Report issues or suggest features using clear and concise titles like `feature_request/include_download_option`.

Please adhere to the Wikimedia Community Universal Code of Conduct when contributing.

## License
WikidataQAChat is open-source software licensed under the MIT License. You are free to use, modify, and distribute the software as you wish. We kindly ask for a citation to this repository if you use WikidataQAChat in your projects.

## Contact
For questions, comments, or discussions, please open an issue on this GitHub repository. We are committed to fostering a welcoming and collaborative community.

![Wikidata and the Meaning of Life](https://github.com/exowanderer/WikidataChat/blob/main/images/WikidataChat_Meaning_of_Life_Graphic.jpg?raw=true)
