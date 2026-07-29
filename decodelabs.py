#decodelabs
def chatbot():
    responses = {
        'hello': 'Hi there! Welcome!',
        'bye': 'See you later!',
        'how are you': "I'm running at full capacity! Thanks for asking.",
        'what is your name': "I'm DecoBot, your rule-based AI assistant!",
        'what can you do': 'I can answer your questions based on my knowledge base!',
        'thanks': "You're welcome! Happy to help.",
        'thank you': "Anytime! That's what I'm here for.",
        'help': "Sure! Try asking me: my name, how I am, or just say hello!",
        'who made you': "I was built by an AI intern at DecodeLabs!",
    }
    while True:  
        raw_input=input()
        clean_input=raw_input.lower().strip()
        if clean_input=='exit':
            break
        reply = responses.get(clean_input, "I'm not sure how to respond to that.")
        print(reply)
chatbot()