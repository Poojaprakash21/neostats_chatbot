from openai import OpenAI 

def chat_with_context(messages, openai_api_key, model="gpt-4o-mini"): 
    client = OpenAI(api_key=openai_api_key) 
    resp = client.chat.completions.create( 
        model=model, 
        messages=messages, 
        max_tokens=800 
    ) 
    return resp.choices[0].message.content