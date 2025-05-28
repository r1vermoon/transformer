from transformers import AutoTokenizer, AutoModelForCausalLM


device = "cuda:0"
prompt="how can i become smarter?"
model_pth="/root/autodl-fs/meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_pth)
model = AutoModelForCausalLM.from_pretrained(
    model_pth,
    device_map=device
)

def chat(prompt):
    input_ids=tokenizer.encode(prompt,return_tensors="pt").to(device=device)
    output=model.generate(        
        input_ids,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

chat_response = chat(prompt)
print(chat_response)
    