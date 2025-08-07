#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <iomanip>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "optimization/nn.hpp"
#include "optimization/activation.hpp"
#include "optimization/kFold.hpp"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// Iris data structure
struct IrisData {
    double sepalLength;
    double sepalWidth;
    double petalLength;
    double petalWidth;
    std::string species;
};

// Load iris data from CSV
std::vector<IrisData> loadIrisData(const std::string& filename) {
    std::vector<IrisData> data;
    std::ifstream file(filename);
    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        IrisData iris;
        
        // Parse CSV line
        std::getline(ss, token, ','); // ID
        std::getline(ss, token, ','); iris.sepalLength = std::stod(token);
        std::getline(ss, token, ','); iris.sepalWidth = std::stod(token);
        std::getline(ss, token, ','); iris.petalLength = std::stod(token);
        std::getline(ss, token, ','); iris.petalWidth = std::stod(token);
        std::getline(ss, token, ','); iris.species = token;
        
        data.push_back(iris);
    }
    
    return data;
}

// Convert species string to class index
int speciesToIndex(const std::string& species) {
    if (species == "Iris-setosa") return 0;
    if (species == "Iris-versicolor") return 1;
    if (species == "Iris-virginica") return 2;
    return -1;
}

// Convert class index to species string
std::string indexToSpecies(int index) {
    switch(index) {
        case 0: return "Iris-setosa";
        case 1: return "Iris-versicolor";
        case 2: return "Iris-virginica";
        default: return "Unknown";
    }
}

// Prepare data for neural network
void prepareData(const std::vector<IrisData>& irisData, 
                Matrix& features, Matrix& targets) {
    features.resize(irisData.size(), 4);
    targets.resize(irisData.size(), 3);
    
    for (size_t i = 0; i < irisData.size(); ++i) {
        // Features: [sepalLength, sepalWidth, petalLength, petalWidth]
        features(i, 0) = irisData[i].sepalLength;
        features(i, 1) = irisData[i].sepalWidth;
        features(i, 2) = irisData[i].petalLength;
        features(i, 3) = irisData[i].petalWidth;
        
        // One-hot encoding for targets
        int classIndex = speciesToIndex(irisData[i].species);
        targets.row(i).setZero();
        targets(i, classIndex) = 1.0;
    }
}

// Calculate accuracy
double calculateAccuracy(const std::vector<int>& predictions, 
                       const std::vector<int>& trueLabels) {
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == trueLabels[i]) correct++;
    }
    return static_cast<double>(correct) / predictions.size();
}

// Calculate precision for a class
double calculatePrecision(const std::vector<int>& predictions, 
                         const std::vector<int>& trueLabels, int classIndex) {
    int truePositives = 0;
    int falsePositives = 0;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == classIndex && trueLabels[i] == classIndex) {
            truePositives++;
        } else if (predictions[i] == classIndex && trueLabels[i] != classIndex) {
            falsePositives++;
        }
    }
    
    return (truePositives + falsePositives > 0) ? 
           static_cast<double>(truePositives) / (truePositives + falsePositives) : 0.0;
}

// Calculate recall for a class
double calculateRecall(const std::vector<int>& predictions, 
                      const std::vector<int>& trueLabels, int classIndex) {
    int truePositives = 0;
    int falseNegatives = 0;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == classIndex && trueLabels[i] == classIndex) {
            truePositives++;
        } else if (predictions[i] != classIndex && trueLabels[i] == classIndex) {
            falseNegatives++;
        }
    }
    
    return (truePositives + falseNegatives > 0) ? 
           static_cast<double>(truePositives) / (truePositives + falseNegatives) : 0.0;
}

// Calculate F1-score
double calculateF1Score(double precision, double recall) {
    return (precision + recall > 0) ? 
           2 * precision * recall / (precision + recall) : 0.0;
}

// Print confusion matrix
void printConfusionMatrix(const std::vector<int>& predictions, 
                         const std::vector<int>& trueLabels) {
    std::cout << "\n=== Confusion Matrix ===" << std::endl;
    std::cout << "Predicted\\Actual\tSetosa\tVersicolor\tVirginica" << std::endl;
    
    for (int predClass = 0; predClass < 3; ++predClass) {
        std::cout << indexToSpecies(predClass) << "\t";
        for (int trueClass = 0; trueClass < 3; ++trueClass) {
            int count = 0;
            for (size_t i = 0; i < predictions.size(); ++i) {
                if (predictions[i] == predClass && trueLabels[i] == trueClass) {
                    count++;
                }
            }
            std::cout << count << "\t";
        }
        std::cout << std::endl;
    }
}

// Print detailed metrics
void printMetrics(const std::vector<int>& predictions, 
                 const std::vector<int>& trueLabels) {
    double accuracy = calculateAccuracy(predictions, trueLabels);
    
    std::cout << "\n=== Classification Metrics ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Overall Accuracy: " << accuracy << " (" << accuracy * 100 << "%)" << std::endl;
    
    std::cout << "\nPer-Class Metrics:" << std::endl;
    std::cout << "Class\t\tPrecision\tRecall\t\tF1-Score" << std::endl;
    
    for (int i = 0; i < 3; ++i) {
        double precision = calculatePrecision(predictions, trueLabels, i);
        double recall = calculateRecall(predictions, trueLabels, i);
        double f1 = calculateF1Score(precision, recall);
        
        std::cout << indexToSpecies(i) << "\t" 
                  << precision << "\t\t" 
                  << recall << "\t\t" 
                  << f1 << std::endl;
    }
    
    printConfusionMatrix(predictions, trueLabels);
}

// K-fold cross validation
void performCrossValidation(const Matrix& features, const Matrix& targets, int k = 5) {
    std::cout << "\n=== K-Fold Cross Validation (k=" << k << ") ===" << std::endl;
    
    std::vector<double> foldAccuracies;
    
    // Create k-fold indices manually to avoid dimension issues
    int foldSize = features.rows() / k;
    std::vector<int> indices(features.rows());
    for (int i = 0; i < features.rows(); ++i) {
        indices[i] = i;
    }
    
    // Shuffle indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    for (int fold = 0; fold < k; ++fold) {
        // Prepare training data (all folds except current)
        std::vector<int> trainIndices, testIndices;
        
        for (int i = 0; i < features.rows(); ++i) {
            if (i >= fold * foldSize && i < (fold + 1) * foldSize) {
                testIndices.push_back(indices[i]);
            } else {
                trainIndices.push_back(indices[i]);
            }
        }
        
        // Create train/test matrices
        Matrix trainFeatures(trainIndices.size(), 4);
        Matrix trainTargets(trainIndices.size(), 3);
        Matrix testFeatures(testIndices.size(), 4);
        Matrix testTargets(testIndices.size(), 3);
        
        for (size_t i = 0; i < trainIndices.size(); ++i) {
            trainFeatures.row(i) = features.row(trainIndices[i]);
            trainTargets.row(i) = targets.row(trainIndices[i]);
        }
        
        for (size_t i = 0; i < testIndices.size(); ++i) {
            testFeatures.row(i) = features.row(testIndices[i]);
            testTargets.row(i) = targets.row(testIndices[i]);
        }
        
        // Train neural network
        neural_network::NeuralNetwork nn({4, 10, 3}, 
                                       activation::ActivationType::Sigmoid, 0.01);
        nn.train(trainFeatures, trainTargets, 100);
        
        // Make predictions
        std::vector<int> predictions, trueLabels;
        for (int i = 0; i < testFeatures.rows(); ++i) {
            Matrix input = testFeatures.row(i);
            Matrix output = nn.predict(input);
            
            // Find predicted class - output is (output_features, samples) = (3, 1)
            int predictedClass = 0;
            output.col(0).maxCoeff(&predictedClass);  // Use col(0) instead of row(0)
            predictions.push_back(predictedClass);
            
            // Find true class
            int trueClass = 0;
            testTargets.row(i).maxCoeff(&trueClass);
            trueLabels.push_back(trueClass);
        }
        
        double foldAccuracy = calculateAccuracy(predictions, trueLabels);
        foldAccuracies.push_back(foldAccuracy);
        
        std::cout << "Fold " << (fold + 1) << " Accuracy: " 
                  << foldAccuracy << " (" << foldAccuracy * 100 << "%)" << std::endl;
    }
    
    // Calculate mean and std of cross-validation accuracy
    double meanAccuracy = 0.0;
    for (double acc : foldAccuracies) {
        meanAccuracy += acc;
    }
    meanAccuracy /= foldAccuracies.size();
    
    double variance = 0.0;
    for (double acc : foldAccuracies) {
        variance += (acc - meanAccuracy) * (acc - meanAccuracy);
    }
    variance /= foldAccuracies.size();
    double stdAccuracy = std::sqrt(variance);
    
    std::cout << "\nCross-Validation Results:" << std::endl;
    std::cout << "Mean Accuracy: " << meanAccuracy << " (" << meanAccuracy * 100 << "%)" << std::endl;
    std::cout << "Std Accuracy: " << stdAccuracy << " (" << stdAccuracy * 100 << "%)" << std::endl;
}

int main() {
    std::cout << "=== Iris Dataset Neural Network Classifier Evaluation ===" << std::endl;
    
    // Load iris data - fix the path to be relative to the executable location
    std::vector<IrisData> irisData = loadIrisData("../../test/data/iris.csv");
    std::cout << "Loaded " << irisData.size() << " iris samples" << std::endl;
    
    if (irisData.empty()) {
        std::cerr << "Error: No data loaded. Please check the file path." << std::endl;
        std::cerr << "Trying alternative paths..." << std::endl;
        
        // Try alternative paths
        std::vector<std::string> paths = {
            "test/data/iris.csv",
            "../test/data/iris.csv",
            "../../test/data/iris.csv",
            "data/iris.csv",
            "../data/iris.csv"
        };
        
        for (const auto& path : paths) {
            std::cout << "Trying path: " << path << std::endl;
            irisData = loadIrisData(path);
            if (!irisData.empty()) {
                std::cout << "Successfully loaded " << irisData.size() << " samples from " << path << std::endl;
                break;
            }
        }
        
        if (irisData.empty()) {
            std::cerr << "Failed to load data from any path. Exiting." << std::endl;
            return 1;
        }
    }
    
    // Prepare data
    Matrix features, targets;
    prepareData(irisData, features, targets);
    
    std::cout << "Features shape: " << features.rows() << "x" << features.cols() << std::endl;
    std::cout << "Targets shape: " << targets.rows() << "x" << targets.cols() << std::endl;
    
    // Split data (80% train, 20% test) with stratified sampling
    std::cout << "Splitting data with stratified sampling..." << std::endl;
    
    // Count samples per class
    std::vector<int> classCounts(3, 0);
    for (int i = 0; i < targets.rows(); ++i) {
        int trueClass = 0;
        targets.row(i).maxCoeff(&trueClass);
        classCounts[trueClass]++;
    }
    
    std::cout << "Class distribution: Setosa=" << classCounts[0] 
              << ", Versicolor=" << classCounts[1] 
              << ", Virginica=" << classCounts[2] << std::endl;
    
    // Create stratified split
    std::vector<int> trainIndices, testIndices;
    for (int classIdx = 0; classIdx < 3; ++classIdx) {
        std::vector<int> classIndices;
        for (int i = 0; i < targets.rows(); ++i) {
            int trueClass = 0;
            targets.row(i).maxCoeff(&trueClass);
            if (trueClass == classIdx) {
                classIndices.push_back(i);
            }
        }
        
        // Shuffle class indices
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(classIndices.begin(), classIndices.end(), g);
        
        // Split this class 80/20
        int trainSize = static_cast<int>(classIndices.size() * 0.8);
        for (int i = 0; i < trainSize; ++i) {
            trainIndices.push_back(classIndices[i]);
        }
        for (int i = trainSize; i < classIndices.size(); ++i) {
            testIndices.push_back(classIndices[i]);
        }
    }
    
    // Create train/test matrices
    Matrix trainFeatures(trainIndices.size(), 4);
    Matrix trainTargets(trainIndices.size(), 3);
    Matrix testFeatures(testIndices.size(), 4);
    Matrix testTargets(testIndices.size(), 3);
    
    for (size_t i = 0; i < trainIndices.size(); ++i) {
        trainFeatures.row(i) = features.row(trainIndices[i]);
        trainTargets.row(i) = targets.row(trainIndices[i]);
    }
    
    for (size_t i = 0; i < testIndices.size(); ++i) {
        testFeatures.row(i) = features.row(testIndices[i]);
        testTargets.row(i) = targets.row(testIndices[i]);
    }
    
    std::cout << "Training samples: " << trainFeatures.rows() << std::endl;
    std::cout << "Test samples: " << testFeatures.rows() << std::endl;
    
    // Verify class distribution in splits
    std::vector<int> trainClassCounts(3, 0), testClassCounts(3, 0);
    for (int i = 0; i < trainTargets.rows(); ++i) {
        int trueClass = 0;
        trainTargets.row(i).maxCoeff(&trueClass);
        trainClassCounts[trueClass]++;
    }
    for (int i = 0; i < testTargets.rows(); ++i) {
        int trueClass = 0;
        testTargets.row(i).maxCoeff(&trueClass);
        testClassCounts[trueClass]++;
    }
    
    std::cout << "Train class distribution: Setosa=" << trainClassCounts[0] 
              << ", Versicolor=" << trainClassCounts[1] 
              << ", Virginica=" << trainClassCounts[2] << std::endl;
    std::cout << "Test class distribution: Setosa=" << testClassCounts[0] 
              << ", Versicolor=" << testClassCounts[1] 
              << ", Virginica=" << testClassCounts[2] << std::endl;
    
    // Train neural network
    std::cout << "\n=== Training Neural Network ===" << std::endl;
    neural_network::NeuralNetwork nn({4, 10, 3}, 
                                   activation::ActivationType::Sigmoid, 0.01);
    nn.train(trainFeatures, trainTargets, 1000);
    
    // Make predictions on test set
    std::vector<int> predictions, trueLabels;
    for (int i = 0; i < testFeatures.rows(); ++i) {
        Matrix input = testFeatures.row(i);
        Matrix output = nn.predict(input);
        
        // Find predicted class - output is (output_features, samples) = (3, 1)
        int predictedClass = 0;
        output.col(0).maxCoeff(&predictedClass);  // Use col(0) instead of row(0)
        predictions.push_back(predictedClass);
        
        // Find true class
        int trueClass = 0;
        testTargets.row(i).maxCoeff(&trueClass);
        trueLabels.push_back(trueClass);
    }
    
    // Print metrics
    printMetrics(predictions, trueLabels);
    
    // Perform cross-validation
    performCrossValidation(features, targets, 5);
    
    std::cout << "\n=== Evaluation Complete ===" << std::endl;
    
    return 0;
}


