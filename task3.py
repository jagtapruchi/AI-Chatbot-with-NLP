import nltk
import tkinter as tk
from tkinter import ttk
from transformers import AutoModelForCausalLM, AutoTokenizer
import random


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")  #tokenizer for token representation of raw text and conerting tokens back to raw text 
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")  #model for including casual language responses by the GPT model

chat_history_ids = None    #recording conversational history. 'None' indicates no history at the start of conversation     

greetings_dataset = ["hello", "hi", "hey", "wassup", "hey dude", "how are you", "howdy"]
greetings_response = ["Hello! How can I help you today?", "Hi there! How can I assist you?", "Hey! What can I do for you?"]

goodbye_dataset = ["thank you", "goodbye", "bye", "good night", "good morning", "good afternoon"]
goodbye_response = ["You're welcome!", "Goodbye! See you soon!", "Bye! Take care!", "Good night!", "Have a great day!"]

#Generate chatbot response
def chatbot_response(user_input):
    global chat_history_ids
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")    #IDs of the tokens

    bot_output = model.generate(
        new_input_ids,
        max_length=1000,                      #maximum no. of tokens in the generated resopnse
        pad_token_id=tokenizer.eos_token_id,  #used for padding the token id when it is <max_length
        do_sample=True,                       #model samples from the probability distribution of possible next tokens rather than always choosing the most likely token
        temperature=0.7,                      #randomness of sampling
        top_p=0.9,                            #model considers the smallest set of tokens whose cumulative probability is greater than or equal to 0.9.
        top_k=50,                             #limits the sampling to the top k(50) most probable next tokens.
    )

    response = tokenizer.decode(bot_output[:, new_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Process user input
def process_input():
    user_message = user_input.get().strip().lower()

    if not user_message:
        return

    chat_history.insert("", tk.END, values=("You", user_message))  #user message into chat history

    if user_message in greetings_dataset:
        bot_reply = random.choice(greetings_response)
    elif user_message in goodbye_dataset:
        bot_reply = random.choice(goodbye_response)
    else:
        bot_reply = chatbot_response(user_message)

    chat_history.insert("", tk.END, values=("Bot", bot_reply))  #bot reply into chat history
    user_input.delete(0, tk.END)

#GUI Design
root = tk.Tk()
root.title("AI Chatbot")
root.geometry("500x600")
root.configure(bg="#f0f8ff")

# Styles
style = ttk.Style()
style.theme_use("clam")
style.configure("Treeview", background="#f8f9fa", foreground="black", rowheight=25, fieldbackground="#f8f9fa", font=("Helvetica", 12))
style.map("Treeview", background=[("selected", "#add8e6")])

style.configure("TButton", font=("Helvetica", 12, "bold"), background="#4682b4", foreground="white")
style.map("TButton", background=[("active", "#5a9bd3")])

# Chat history treeview
columns = ("User", "Message")
chat_history = ttk.Treeview(root, columns=columns, show="headings", height=20)
chat_history.heading("User", text="User")
chat_history.heading("Message", text="Message")
chat_history.column("User", width=100, anchor="center")
chat_history.column("Message", width=380, anchor="w")
chat_history.pack(pady=10, padx=10)

# Input frame
frame = tk.Frame(root, bg="#f0f8ff")
frame.pack(pady=10)

# User input field
user_input = ttk.Entry(frame, width=40, font=("Helvetica", 14))
user_input.pack(side=tk.LEFT, padx=5)

# Send button
send_button = ttk.Button(frame, text="Send", command=process_input)
send_button.pack(side=tk.LEFT)

# Footer
footer = tk.Label(root, text="Powered by DialoGPT", bg="#f0f8ff", fg="gray", font=("Helvetica", 10))
footer.pack(pady=10)

root.mainloop()
