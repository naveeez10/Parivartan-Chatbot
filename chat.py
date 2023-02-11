import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "viz"


def get_response(msg):
    if msg == '1':
        return "You can add funds to metamask using any methods out of the following: Credit Cards, Debit Cards, UPI, and Net Banking"
    elif msg == '2':
        return "You can withdraw funds directly to your linked bank account."
    elif msg == '3':
        return "Bank accounts are centralized, i.e., if the bank fails, you lose your funds. It has a single authority that is also a point of failure. In contrast, wallets are decentralized and provide better security."
    elif msg == '4':
        return "To connect any wallet, you can just add the said wallet's extension to your browser. After that, you can click on the 'Connect Wallet' button the homepage to connect your wallet"
    elif msg == '5':
        return "When you set up your wallet, you are given a 12 word security phrase which you should only keep to yourself. It can be used to retrieve the account in case it's lost"

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand...Here are some questions i am asked often: \n 1. How to add funds to my wallet? \n 2. How to withdraw funds from my wallet? \n 3. Why can't I use my bank account instead of a wallet? \n 4. How to connect my wallet? \n 5. How do you secure a wallet?. \n Choose one of the above numbers or ask any other questions you have"


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
