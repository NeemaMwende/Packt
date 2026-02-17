from transformers import pipeline

summerizer = pipeline(task="summarization", model="facebook/bart-large-cnn")

text = "Transformers are deep learning models designed to handle sequential data using a mechanism called self-attention. Unlike recurrent networks, they process entire sequences in parallel, making them faster and more scalable. Introduced in the 2017 paper Attention Is All You Need, transformers power modern language models, enabling tasks such as translation, summarization, and code generation with remarkable accuracy across diverse domains."

summary= summerizer(text, max_length=150, clean_up_tokenization_spaces=True)
print(summary[0]["summary_text"])