- MLP is a type of feed-forward neural network
  - Each of its layers is a fixed size
  - Each layer's output is fed into the next layer as input
  - While feed-forward neural networks are great for tasks involving fixed size input data
  - They aren't as great in dealing with sequences of text data


- Recurrent neural networks (RNN)
  - Specially designed to work with sequential data of varying lengths
  - The main component of a recurrent neural network (RNN) is its cell

- Two ways to look at an RNN
  - Rolled 
  - Unrolled

- Rolled RUN, "true" depiction of the network:
  - It consists of a single cell 
  - multi-layer RNN will have multi-stacked cells
  - 3 types of connections: input, output, and recurrent 
  - An unrolled representation gives us a better look at what each of these connections represent

- Unrolled representation:
  - The RNN consists of 3 time steps
  - Length of the RNN's input sequence is 3
  - Meaning RNN will output a sequence of length 3
  - At each time step, the arrow going into the cell represents the token at that particular index of the input sequence
  - The arrow going out of the cell represents the cell's output
