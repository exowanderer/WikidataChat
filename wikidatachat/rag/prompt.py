

def get_prompt():

    prompt_template = """
    Given the information in the triple backticks below, are you able to answer
    the following question:

    Do not answer the question. Only respond with 'yes' or 'no' if you are able to answer the question.

    Question: {question}.

    Context: {context}
    """

    return PromptTemplate.from_template(prompt_template)
