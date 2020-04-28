# AI Neural Network
Artificial Intelligence that uses neural network learning tecnic for supervised classification.

<h3>Sample Input Format</h3>

The first line contains 7 numbers separated by space <b>(IN, HN, ON, LR, M, S1, S2)</b>
1. <b>IN</b> = Input length of the neural network (Integer)
2. <b>HN</b> = Number of hidden neurons (Integer)
3. <b>ON</b> = Number of possible outputs (Integer)
4. <b>LR</b> = Learning rate (float)
5. <b>M</b> = Momentum (float)
6. <b>S1</b> = Size of the training set (Integer)
7. <b>S2</b> = Size of the test set (Integer)

The second line contains <b>ON</b> strings, corresponding to the possible outputs.

In each of the next <b>S1+S2</b> lines will contains <b>IN</b> numbers followed by a string as follow:
1. <b>IN</b> numbers, representing the input data
2. Expected output

Run the command `g++ Main.cpp Include/AI/NeuralNet.cpp -o AINeuralNetwork.exe` to compile the code, and then run the command `./AINeuralNetwork.exe < Sample.txt` to test with the Sample
