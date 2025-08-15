#include "SimpleCNN.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>  // std::shuffle
#include <random>     // std::random_device, std::mt19937

void train_from_csv(SimpleCNN& model, const std::string& csv_path, int epochs, double lr) {
    // Load dataset once
    std::vector<std::pair<std::string, int>> dataset;

    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << csv_path << std::endl;
        return;
    }

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string img_path;
        int label;

        if (!std::getline(ss, img_path, ',')) continue;
        if (!(ss >> label)) continue;

        dataset.emplace_back(img_path, label);
    }

    if (dataset.empty()) {
        std::cerr << "No data found in CSV." << std::endl;
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle dataset every epoch
        std::shuffle(dataset.begin(), dataset.end(), gen);

        double total_loss = 0.0;
        int correct = 0;

        for (const auto& [img_path, label] : dataset) {
            Eigen::Tensor<double, 3> input_tensor = model.convert_images(img_path);
            auto [prob, loss] = model.forward_pass(input_tensor, label);

            total_loss += loss;

            int predicted = 0;
            prob.maxCoeff(&predicted);
            if (predicted == label) ++correct;

            model.back_pass(lr);
        }

        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << " - Loss: " << total_loss / dataset.size()
                  << " - Accuracy: " << (double)correct / dataset.size() * 100 << "%" << std::endl;
    }
}

int main() {
    SimpleCNN model;
    std::string csv_path = "/home/vishal/Vikash/Projects/Final_project/image_labels.csv";
    int epochs = 10;
    double learning_rate = 0.01;

    train_from_csv(model, csv_path, epochs, learning_rate);
    model.save("simple_cnn_model.txt");
    return 0;
}
