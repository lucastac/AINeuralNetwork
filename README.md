# AI Neural Network
Artificial Intelligence that uses neural network learning tecnic for supervised classification.

<h3>Sample Input Format</h3>

The first lines contains 7 numbers separated by space <b>(IN, HN, ON, LR, M, S1, S2)</b>
1. <b>IN</b> = Input length of the neural network (Integer)
2. <b>HN</b> = Number of hidden neurons (Integer)
3. <b>ON</b> = Number of possible outputs (Integer)
4. <b>LR</b> = Learning rate (float)
5. <b>M</b> = Momentum (float)
6. <b>S1</b> = Size of the training set (Integer)
7. <b>S2</b> = Size of the test set (Integer)

In each of the next S1+S2 lines will contains IN+ON numbers as follow:
1. First IN numbers, represents the input data
2. Next ON numbers with 0s and 1s informing the expected ouput

Run the command `g++ Main.cpp Include/AI/NeuralNet.cpp -o AINeuralNetwork.exe` to compile the code, and then run the command `./AINeuralNetwork.exe < Sample.txt` to test with the Sample
