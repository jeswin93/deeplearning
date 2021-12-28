import math

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import  *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        # input to hidden
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # input to output
        self.i2o = nn.Linear(input_size +  hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor,  hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        output_nn = self.i2o(combined)
        output_nn = self.softmax(output_nn)
        return output_nn, hidden

    def  init_hidden(self):
        return torch.zeros(1, self.hidden_size)


# category_lines is a dict  with names and corresponding country  as key
# all_categories is a list  of  all countries
category_lines, all_categories = load_data()
n_categories = len(all_categories)

# hyperparameter number of units in hidden layer
n_hidden = 128
rnn = RNN(N_LETTERS, n_hidden,  n_categories)

# Negative Log Likelihood loss
criterion = nn.NLLLoss()
learning_rate = 0.003
optimizer = torch.optim.SGD(rnn.parameters(), learning_rate)


def train_step(line_tensor,  category_tensor):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output,  category_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

current_loss=0
all_losses = []
plot_steps, print_steps = 1000, 5000

n_iters = 200000

def  category_from_output(output_tensor):
    category_idx = torch.argmax(output_tensor)
    return all_categories[category_idx]

for i in range(n_iters):
    category,  line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)

    output, loss = train_step(line_tensor, category_tensor)
    current_loss += loss

    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss/plot_steps)
        current_loss = 0

    if (i+1) %  print_steps == 0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(f"{i + 1}, {(i+1)/n_iters*100} {loss:.4f} {line} / {guess}  {correct}")

plt.figure()
plt.plot(all_losses)
plt.show()

def predict(input_line):
    with  torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        hidden  = rnn.init_hidden()
        for i in range(line_tensor.size()[0]):
            ouput, hidden = rnn(line_tensor,hidden)

        category  = category_from_output(output)

        print(f"{input_line}, {category}")


while True:
    line = input("Input name:")
    if line == 'quit':
        break

    predict(line)