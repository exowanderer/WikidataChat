from haystack.components.builders.prompt_builder import PromptBuilder

prompt_template_en = """
Documents:
{% for doc_ in documents %}
    {{ doc_.content }}
{% endfor %}
With this documents, answer the following question: {{question}}
"""

prompt_template_de = """
Dokumente:
{% for doc_ in documents %}
    {{ doc_.content }}
{% endfor %}
Mit diesen Dokumenten, beantworte die folgende Frage: {{question}}
"""

system_prompts = {
    'en': 'You are a helpful assistant. You answer questions based on the given documents. Answer based on the documents only. If the information is not in the documents, say that you cannot find the information.',
    'de': 'Du bist ein hilfreicher Assistent. Du beantwortest Fragen basierend auf den vorliegenden Dokumenten. Beantworte basierend auf den Dokumenten nur. Wenn die Information nicht in den Dokumenten ist, sage, dass du sie nicht finden kannst.',
}

user_prompt_builders = {
    'en': PromptBuilder(template=prompt_template_en),
    'de': PromptBuilder(template=prompt_template_de),
}

