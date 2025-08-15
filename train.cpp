#include "SimpleCNN.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>  // For std::shuffle
#include <random>     // For random device and generator

// ======================= TRAINING FUNCTION =======================
void train_from_csv(SimpleCNN& model, const std::string& csv_path, int epochs, double lr) {
    // Load the dataset once into memory
    std::vector<std::pair<std::string, int>> dataset;

    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << csv_path << std::endl;
        return;
    }

    std::string line;
    std::getline(file, line); // Skip header line

    // Parse each line: first column = image path, second column = label
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string img_path;
        int label;

        if (!std::getline(ss, img_path, ',')) continue; // get image path
        if (!(ss >> label)) continue;                   // get label

        dataset.emplace_back(img_path, label);
    }

    if (dataset.empty()) {
        std::cerr << "No data found in CSV." << std::endl;
        return;
    }

    // Random generator for shuffling
    std::random_device rd;
    std::mt19937 gen(rd());

    // Loop over epochs
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle dataset at the beginning of each epoch
        std::shuffle(dataset.begin(), dataset.end(), gen);

        double total_loss = 0.0;
        int correct = 0;

        // Train on each sample
        for (const auto& [img_path, label] : dataset) {
            // Convert image to normalized tensor
            Eigen::Tensor<double, 3> input_tensor = model.convert_images(img_path);

            // Forward pass
            auto [prob, loss] = model.forward_pass(input_tensor, label);
            total_loss += loss;

            // Compute prediction
            int predicted = 0;
            prob.maxCoeff(&predicted);
            if (predicted == label) ++correct;

            // Backward pass and parameter update
            model.back_pass(lr);
        }

        // Print epoch statistics
        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << " - Loss: " << total_loss / dataset.size()
                  << " - Accuracy: " << (double)correct / dataset.size() * 100 << "%" << std::endl;
    }
}

// ======================= MAIN FUNCTION ===========================
int main() {
    SimpleCNN model; // Initialize CNN model

    std::string csv_path = "/home/vishal/Vikash/Projects/Final_project/image_labels.csv";
    int epochs = 10;           // Number of training epochs
    double learning_rate = 0.01; // Learning rate for SGD

    train_from_csv(model, csv_path, epochs, learning_rate);

    // Save trained model to disk
    model.save("simple_cnn_model.txt");

    return 0;
}
