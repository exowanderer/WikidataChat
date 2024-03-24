
def bloom_pipeline(
        prompt, question='', context='', max_length=2048, n_sequences=1, quantization=True,
        model_name="bigscience/bloom-560m"):

    model, tokenizer = initialize_model_and_tokenizer(
        model_name=model_name,
        quantization=quantization
    )

    # print(f'{len(tokenizer.encode(prompt))=}')

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=10
    )
    hf = HuggingFacePipeline(pipeline=pipe)

    chain = prompt | hf

    return chain.invoke({"question": question, "context": context})

    # print(result)

    # return result
    """
    generated_texts = generate_text(
        prompt,
        model,
        tokenizer,
        max_length=max_length,
        n_sequences=n_sequences
    )
    print(generated_texts)
    return generated_texts
    """


def initialize_model_and_tokenizer(
        model_name="bigscience/bloom-560m", quantization=True):

    quantization_config = None
    if quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # llm_int8_threshold=200.0,
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     device_map=0,
    #     load_in_4bit=True
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BloomForQuestionAnswering.from_pretrained(
        model_name,
        torch_dtype=None if device.type == 'cpu' else torch.float16,
        device_map="auto" if device.type == 'cpu' else 0,
        quantization_config=quantization_config,
        # load_in_8bit_fp32_cpu_offload=True
    )

    # model = model.to(device)

    return model, tokenizer


def generate_text(prompt, model, tokenizer, max_length=2048, n_sequences=1):
    # Encode the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)

    # Generate outputs
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        # num_max_sequences=n_sequences
    )

    # Decode the outputs to text
    return [
        tokenizer.decode(
            output,
            skip_special_tokens=True
        )
        for output in outputs
    ]
