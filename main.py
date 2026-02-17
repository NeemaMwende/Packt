# text summarization
from transformers import pipeline
# summerizer = pipeline(task="summarization", model="facebook/bart-large-cnn")
# text = "Transformers are deep learning models designed to handle sequential data using a mechanism called self-attention. Unlike recurrent networks, they process entire sequences in parallel, making them faster and more scalable. Introduced in the 2017 paper Attention Is All You Need, transformers power modern language models, enabling tasks such as translation, summarization, and code generation with remarkable accuracy across diverse domains."
# summary= summerizer(text, max_length=150, clean_up_tokenization_spaces=True)
# print(summary[0]["summary_text"])


# text generation
# generator = pipeline(task="text-generation", model="distilgpt2")
# review = "This book was great. I enjoyed the plot twist in chapter 2"
# response = "Dear reader thank you for your review"
# prompt = f"Book review:\n{review}\n\nBook shop response to the review:\n{response}"
# # prompt = "The kenyan government is famous for"
# output = generator(prompt, max_length=100, pad_token_id=generator.tokenizer.eos_token_id, truncation=True)
# print(output[0]["generated_text"])


#text translation
translator = pipeline(task="translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
text = "I dont know how i am going to do it yet but i know i will get there"
output = translator(text, clean_up_tokenization_spaces=True)
print(output[0]["translation_text"])

