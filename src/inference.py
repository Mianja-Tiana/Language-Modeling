from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(model_path):
    return GPT2LMHeadModel.from_pretrained(model_path)


def load_tokenizer(tokenizer_path):
    return GPT2Tokenizer.from_pretrained(tokenizer_path)


def generate_text(sequence, max_length, model_path="models/result"):
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    sequence = input("Prompt : ")
    max_len = int(input("Longueur max : "))
    generate_text(sequence, max_len)
