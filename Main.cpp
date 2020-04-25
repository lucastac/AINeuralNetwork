#include <iostream>
#include <iomanip>
#include "Include/AI/NeuralNet.h"

using namespace std;

#define seenPrecision 5

int main()
{
    int entryNumber, hiddenNumber, outPutNumber, trainingSize, testSize;
    double learningRate, momentum;

    cin >> entryNumber >> hiddenNumber >> outPutNumber >> learningRate >> momentum >> trainingSize >> testSize;

    cout << std::fixed << std::setprecision(seenPrecision);

    NeuralNet NT(entryNumber, hiddenNumber, outPutNumber, learningRate, momentum);

    vector< vector<double> > EntryToLearn;
    vector< vector<double> > ResultToLearn;

    for (int i = 0; i < trainingSize; i++)
    {
        vector<double> entry;
        vector<double> expectedResult;

        for (int j = 0; j < entryNumber; j++)
        {
            double ent;
            cin >> ent;
            entry.push_back(ent);
        }
        for (int j = 0; j < outPutNumber; j++)
        {
            double res;
            cin >> res;
            expectedResult.push_back(res);
        }
        EntryToLearn.push_back(entry);
        ResultToLearn.push_back(expectedResult);
    }

    for (int i = 0; i < 1000; i++)
    {
        for (int j = 0; j < trainingSize; j++)
        {
            NT.LearnEntry(EntryToLearn[j], ResultToLearn[j]);
        }
    }


    NT.ShowWeights();
    cout << endl;
    for (int i = 0; i < testSize; i++)
    {
        vector<double> entry;
        vector<double> expectedResult;

        for (int j = 0; j < entryNumber; j++)
        {
            double ent;
            cin >> ent;
            entry.push_back(ent);
        }
        for (int j = 0; j < outPutNumber; j++)
        {
            double res;
            cin >> res;
            expectedResult.push_back(res);
        }

        vector< vector<double> > res = NT.ResultClassification(entry);
        cout << "Test " << i + 1 << " :" << endl << endl;
        for (int j = 0; j < res[1].size(); j++)
        {
            cout << j + 1 << " Res : " << res[1][j] << " Exp : " << expectedResult[j] << endl;
        }
        cout << endl;

    }
}
