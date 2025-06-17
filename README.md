# CodeAlpha_iris-classification-of-flowers
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include <ctime>

using namespace std;

struct Iris {
    vector<float> features;
    string label;
};

float euclideanDistance(const vector<float>& a, const vector<float>& b) {
    float sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += pow(a[i] - b[i], 2);
    return sqrt(sum);
}

string classifyKNN(const vector<Iris>& trainingSet, const vector<float>& testInstance, int k) {
    vector<pair<float, string>> distances;

    for (const auto& iris : trainingSet) {
        float dist = euclideanDistance(iris.features, testInstance);
        distances.push_back({dist, iris.label});
    }

    sort(distances.begin(), distances.end());

    map<string, int> votes;
    for (int i = 0; i < k; ++i)
        votes[distances[i].second]++;

    string predictedLabel;
    int maxVotes = 0;
    for (const auto& vote : votes) {
        if (vote.second > maxVotes) {
            maxVotes = vote.second;
            predictedLabel = vote.first;
        }
    }
    return predictedLabel;
}

vector<Iris> loadEmbeddedDataset() {
    vector<Iris> dataset;
    string data = R"(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
6.2,2.2,4.5,1.5,Iris-versicolor
5.9,3.2,4.8,1.8,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
7.1,3.0,5.9,2.1,Iris-virginica
5.0,3.6,1.4,0.2,Iris-setosa
6.0,2.2,4.0,1.0,Iris-versicolor
6.7,3.1,4.7,1.5,Iris-versicolor
6.5,3.0,5.8,2.2,Iris-virginica
)";

    istringstream ss(data);
    string line;
    getline(ss, line); // Skip header

    while (getline(ss, line)) {
        Iris iris;
        istringstream linestream(line);
        string val;
        int col = 0;
        while (getline(linestream, val, ',')) {
            if (col < 4) iris.features.push_back(stof(val));
            else iris.label = val;
            col++;
        }
        dataset.push_back(iris);
    }

    return dataset;
}

void splitDataset(const vector<Iris>& fullSet, vector<Iris>& trainSet, vector<Iris>& testSet, float testRatio = 0.3) {
    vector<Iris> shuffled = fullSet;
    srand(time(0));
    random_shuffle(shuffled.begin(), shuffled.end());

    size_t testSize = static_cast<size_t>(shuffled.size() * testRatio);
    testSet.assign(shuffled.begin(), shuffled.begin() + testSize);
    trainSet.assign(shuffled.begin() + testSize, shuffled.end());
}

int main() {
    vector<Iris> fullData = loadEmbeddedDataset();
    vector<Iris> trainSet, testSet;
    splitDataset(fullData, trainSet, testSet);

    int k = 3;
    int correct = 0;

    for (const auto& testInstance : testSet) {
        string predicted = classifyKNN(trainSet, testInstance.features, k);
        cout << "Actual: " << testInstance.label << ", Predicted: " << predicted << endl;
        if (predicted == testInstance.label)
            correct++;
    }

    float accuracy = (float)correct / testSet.size() * 100;
    cout << "\nAccuracy: " << accuracy << "% on " << testSet.size() << " test samples.\n";

    return 0;
}
