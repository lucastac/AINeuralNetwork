#include <iostream>
#include <cmath>
#include "NeuralNet.h"

// random value 0 - 1 for initial weights
double NeuralNet::randomWeight() { return rand() / double(RAND_MAX); }

double NeuralNet::Sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double NeuralNet::SigmoidDerivate(double x)
{
    double ex = exp(-x);
    return ex / pow(1 + ex, 2);
}

NeuralNet::NeuralNet(int entryNumber, int hiddenNumber, int outPutNumber, double learningRate, double momentum)
{
    //Initialize the weights between Entry and Hidden layer
    EntryHiddenWeights = vector< vector< double > >(hiddenNumber);
    EntryHiddenWeights_OldDelta = vector< vector< double > >(hiddenNumber);
    for (int i = 0; i < hiddenNumber; i++)
    {
        for (int j = 0; j < entryNumber; j++)
        {
            EntryHiddenWeights[i].push_back(randomWeight());
            EntryHiddenWeights_OldDelta[i].push_back(0);
        }

    }

    //Initialize the weights between Hidden and Out layer
    HiddenOutWeights = vector< vector< double > >(outPutNumber);
    HiddenOutWeights_OldDelta = vector< vector< double > >(outPutNumber);
    for (int i = 0; i < outPutNumber; i++)
    {
        for (int j = 0; j < hiddenNumber; j++)
        {
            HiddenOutWeights[i].push_back(randomWeight());
            HiddenOutWeights_OldDelta[i].push_back(0);
        }

    }

    LearningRate = learningRate;
    Momentum = momentum;
}

//calculates the result classification for an entry {hidden result, output result}
vector< vector<double> > NeuralNet::ResultClassification(const vector<double>& entry)
{
    vector<double> hiddenValues(HiddenOutWeights[0].size());
    vector<double> OutPut(HiddenOutWeights.size());
    //calculates the return for the hidden Layer
    for (int i = 0; i < EntryHiddenWeights.size(); i++)
    {
        double sumValue = 0;
        for (int j = 0; j < EntryHiddenWeights[i].size(); j++)
        {
            sumValue += EntryHiddenWeights[i][j] * entry[j]; //sum entry with weights
        }

        hiddenValues[i] = Sigmoid(sumValue); //sigmoid function
    }

    //calculates the return for the Output Layer
    for (int i = 0; i < HiddenOutWeights.size(); i++)
    {
        double sumValue = 0;
        for (int j = 0; j < HiddenOutWeights[i].size(); j++)
        {
            sumValue += HiddenOutWeights[i][j] * hiddenValues[j]; //sum entry with weights
        }

        OutPut[i] = Sigmoid(sumValue); //sigmoid function
    }

    vector< vector<double> > result;
    result.push_back(hiddenValues);
    result.push_back(OutPut);

    return result;
}

//calculates the result and change neuralNet based on the error
void NeuralNet::LearnEntry(const vector<double>& entry, const vector<double>& expectedResult)
{
    vector< vector<double> > netResult = ResultClassification(entry);
    vector<double> Error(netResult[1].size());

    //calculates the error
    for (int i = 0; i < expectedResult.size(); i++)
    {
        Error[i] = expectedResult[i] - netResult[1][i];
        if (abs(Error[i]) <= 0.01)
            Error[i] = 0;
    }

    vector< vector< double > > EntryHiddenWeights_newWeight(netResult[0].size()); //new weights between Entry and Hidden layer
    vector< vector< double > > HiddenOutWeights_newWeight(netResult[1].size());//new weights between Hidden and Out layer

    vector< vector< double > > EntryHiddenWeights_newDelta(netResult[0].size()); //new Delta weights between Entry and Hidden layer
    vector< vector< double > > HiddenOutWeights_newDelta(netResult[1].size());//new Delta weights between Hidden and Out layer

    //sigma values for the hidden layer and output layer
    vector<double> sigmaOutputs(netResult[1].size());
    vector<double> sigmaHiddens(netResult[0].size());

    //calculate sigma for the output layer
    for (int i = 0; i < sigmaOutputs.size(); i++)
    {
        sigmaOutputs[i] = Error[i] * SigmoidDerivate(netResult[1][i]);
    }

    //calculates new weights for Hidden -> Output layer
    for (int i = 0; i < HiddenOutWeights.size(); i++)
    {
        for (int j = 0; j < HiddenOutWeights[i].size(); j++)
        {
            double hiddenInfo = netResult[0][j];
            // if(netResult[0][j]>=0.1 || netResult[0][j]<=-0.1)
              //   hiddenInfo = 1/netResult[0][j];

            double newDelta = HiddenOutWeights_OldDelta[i][j] * Momentum + 2 * LearningRate * sigmaOutputs[i] * hiddenInfo;
            HiddenOutWeights_newDelta[i].push_back(newDelta);
            HiddenOutWeights_newWeight[i].push_back(HiddenOutWeights[i][j] + newDelta);
        }

    }

    //calculate sigma for hidden layer
    for (int i = 0; i < sigmaHiddens.size(); i++)
    {
        sigmaHiddens[i] = 0;
        for (int j = 0; j < sigmaOutputs.size(); j++)
        {
            sigmaHiddens[i] += HiddenOutWeights[j][i] * sigmaOutputs[j];
        }
        sigmaHiddens[i] *= SigmoidDerivate(netResult[0][i]);

    }

    //calculates new weights for Entry -> Hidden layer
    for (int i = 0; i < EntryHiddenWeights.size(); i++)
    {
        for (int j = 0; j < EntryHiddenWeights[i].size(); j++)
        {
            double entryInfo = entry[j];
            //if(entry[j] >= 0.1 || entry[j] <= -0.1)
            //    entryInfo = 1/entry[j];

            double newDelta = EntryHiddenWeights_OldDelta[i][j] * Momentum + LearningRate * sigmaHiddens[i] * entryInfo;
            EntryHiddenWeights_newDelta[i].push_back(newDelta);
            EntryHiddenWeights_newWeight[i].push_back(EntryHiddenWeights[i][j] + newDelta);
        }
    }


    //Update the new values of weight and delta do the neuralNet
    for (int i = 0; i < EntryHiddenWeights.size(); i++)
    {
        for (int j = 0; j < EntryHiddenWeights[i].size(); j++)
        {
            if (EntryHiddenWeights_newWeight[i][j] != EntryHiddenWeights_newWeight[i][j])
                return;
            EntryHiddenWeights[i][j] = EntryHiddenWeights_newWeight[i][j];
            EntryHiddenWeights_OldDelta[i][j] = EntryHiddenWeights_newDelta[i][j];
        }
    }

    for (int i = 0; i < HiddenOutWeights.size(); i++)
    {
        for (int j = 0; j < HiddenOutWeights[i].size(); j++)
        {
            HiddenOutWeights[i][j] = HiddenOutWeights_newWeight[i][j];
            HiddenOutWeights_OldDelta[i][j] = HiddenOutWeights_newDelta[i][j];
        }
    }

}

//Print on screen the weights values
void NeuralNet::ShowWeights()
{
    cout << "Entry - Hidden weights" << endl;
    for (int i = 0; i < EntryHiddenWeights.size(); i++)
    {
        for (int j = 0; j < EntryHiddenWeights[i].size(); j++)
        {
            cout << EntryHiddenWeights[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    cout << "Hidden - Out weights" << endl;
    for (int i = 0; i < HiddenOutWeights.size(); i++)
    {
        for (int j = 0; j < HiddenOutWeights[i].size(); j++)
        {
            cout << HiddenOutWeights[i][j] << " ";
        }
        cout << endl;
    }
}
