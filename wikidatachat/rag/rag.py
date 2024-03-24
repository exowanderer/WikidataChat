from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


def question_answer_pipeline(
        question, max_length=2048, n_sequences=1,
        model_name="bigscience/bloom-560m", serapi_api_key=None,
        lang='en', quantization=True, timeout=100, n_cores=cpu_count(),
        text_input=None, api_url='https://www.wikidata.org/w',
        wikidata_base='"wikidata.org"', verbose=False):

    # question = "When was coffee invented?"
    if serapi_api_key is None:
        serapi_api_key = SERAPI_API_KEY

    if text_input is None:
        text_input = get_wikidata_statements_from_query(
            question=question,
            lang=lang,
            timeout=timeout,
            n_cores=n_cores,
            wikidata_base='"wikidata.org"',
            serapi_api_key=serapi_api_key,
            verbose=verbose
        )

    prompt = get_prompt()

    # Example Usage
    bloom_generated_texts = bloom_pipeline(
        prompt=prompt,
        question=question,
        context=text_input,
        max_length=max_length,
        n_sequences=n_sequences,
        model_name=model_name,
        quantization=quantization
    )

    print(f'{len(bloom_generated_texts)=}')

    print('\n Bloom Text Results:')
    for bloom_text_ in bloom_generated_texts:
        print(bloom_text_)
