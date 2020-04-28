#pragma once
#include <vector>

using namespace std;

class NeuralNet {

private:
    vector< vector< double > > EntryHiddenWeights; //weights between Entry and Hidden layer
    vector< vector< double > > HiddenOutWeights;//weights between Hidden and Out layer

    //used with momentum to avoid the net from forget what it learned
    vector< vector< double > > EntryHiddenWeights_OldDelta; //old delta weights between Entry and Hidden layer
    vector< vector< double > > HiddenOutWeights_OldDelta;//old delta weights between Hidden and Out layer

    vector<string> Outputs; //all possible outputs

    double LearningRate;//Learning rate
    double Momentum;//momentum


    static double randomWeight(void); // random value 0 - 1 for initial weights

    double Sigmoid(double x);

    double SigmoidDerivate(double x);

    //calculates the result classification for an entry {hidden result, output result}
    vector< vector<double> > ClassifyForLearn(const vector<double>& entry);

public:
    NeuralNet(int entryNumber, int hiddenNumber, const vector<string>& possibleOutputs, double learningRate, double momentum); // Constructor

    //calculates the result classification for an entry {string result, double certainty level}
    pair<string, double> Classify(const vector<double>& entry);

    //calculates the result and change neuralNet based on the error
    void LearnEntry(const vector<double>& entry, const string& expectedResult);

    //Print on screen the weights values
    void ShowWeights();
};

