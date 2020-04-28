#include <iostream>
#include <iomanip>
#include "Include/AI/NeuralNet.h"

using namespace std;

#define seenPrecision 5

int main()
{
    int entryNumber, hiddenNumber, outPutNumber, trainingSize, testSize;
    double learningRate, momentum;

    // read informations of the neural net
    cin >> entryNumber >> hiddenNumber >> outPutNumber >> learningRate >> momentum >> trainingSize >> testSize;

    vector<string> outputs(outPutNumber);

    // read the possible outputs
    for (int i = 0; i < outPutNumber; i++)
    {
        cin >> outputs[i];
    }

    cout << std::fixed << std::setprecision(seenPrecision);

    NeuralNet NT(entryNumber, hiddenNumber, outputs, learningRate, momentum);

    vector< vector<double> > EntryToLearn;
    vector< string > ResultToLearn;

    // read the training inputs
    for (int i = 0; i < trainingSize; i++)
    {
        vector<double> entry;
        string expectedResult;

        for (int j = 0; j < entryNumber; j++)
        {
            double ent;
            cin >> ent;
            entry.push_back(ent);
        }

        cin >> expectedResult;

        EntryToLearn.push_back(entry);
        ResultToLearn.push_back(expectedResult);
    }

    // train the neural network
    for (int i = 0; i < 1000; i++)
    {
        for (int j = 0; j < trainingSize; j++)
        {
            NT.LearnEntry(EntryToLearn[j], ResultToLearn[j]);
        }
    }

    // print the layers weights after finishing the training;
    NT.ShowWeights();
    cout << endl;

    // read the test inputs
    for (int i = 0; i < testSize; i++)
    {
        vector<double> entry;
        string expectedResult;

        for (int j = 0; j < entryNumber; j++)
        {
            double ent;
            cin >> ent;
            entry.push_back(ent);
        }

        cin >> expectedResult;

        pair<string, double> res = NT.Classify(entry);

        // print the test result
        cout << "Test " << i + 1 << endl;
        cout << "--Result   : " << res.first << endl; 
        cout << "--Expected : " << expectedResult << endl;
        cout << "--Certainty: " << (res.second * 100) << "%" << endl;
        cout << endl;
    }
}
