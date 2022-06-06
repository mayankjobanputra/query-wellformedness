from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


tokenizer = AutoTokenizer.from_pretrained("allenai/macaw-large")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/macaw-large")
input_string = "$sentence$ = Sky is blue; $answer$ = $Blue$ ; $question$ = "
# input_string = "if $sentence$ is: My name is Mayank, and the $answer$ is: Mayank; the question should the $question$ be? "

input_ids = tokenizer.encode(input_string, return_tensors="pt")
output = model.generate(input_ids, max_length=200)

tokenizer.batch_decode(output, skip_special_tokens=True)