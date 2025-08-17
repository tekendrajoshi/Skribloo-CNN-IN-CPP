#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>

// Include your CNN headers here
#include "SimpleCNN.hpp"
#include "helpers.hpp"

// Color definitions for modern styling
struct Color {
    uint8_t r, g, b, a;
    Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) : r(r), g(g), b(b), a(a) {}
};

// Modern color palette
namespace Colors {
    const Color BACKGROUND(25, 25, 35);
    const Color SURFACE(40, 44, 52);
    const Color PRIMARY(64, 123, 255);
    const Color PRIMARY_HOVER(84, 143, 255);
    const Color SUCCESS(34, 197, 94);
    const Color ERROR(239, 68, 68);
    const Color WARNING(245, 158, 11);
    const Color TEXT_PRIMARY(248, 250, 252);
    const Color TEXT_SECONDARY(148, 163, 184);
    const Color ACCENT(139, 92, 246);
    const Color CANVAS_BG(255, 255, 255);
    const Color CANVAS_BORDER(203, 213, 225);
}

class SkriblooAIGUI {
private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    TTF_Font* font;
    TTF_Font* titleFont;
    TTF_Font* smallFont;
    
    // CNN Model
    SimpleCNN model;
    
    // Game state
    enum GameState { MENU, DRAWING, RESULT };
    GameState currentState;
    
    // Drawing canvas
    static const int CANVAS_SIZE = 280;
    static const int CANVAS_X = 160;
    static const int CANVAS_Y = 120;
    
    // Window dimensions
    static const int WINDOW_WIDTH = 800;
    static const int WINDOW_HEIGHT = 600;
    
    // Drawing data
    std::vector<std::vector<bool>> canvas;
    bool isDrawing;
    
    // Animation variables
    float animationTime;
    int pulsePhase;
    
    // Word categories
    std::vector<std::string> categories = {
        "airplane", "apple", "bicycle", "book", "car",
        "cat", "chair", "clock", "cloud", "cup"
    };
    std::string prediction;
    std::string secondPrediction;
    std::string thirdPrediction;
    float confidence;
    float secondConfidence;
    float thirdConfidence;
    
    // Modern styled buttons
    struct Button {
        SDL_Rect rect;
        std::string text;
        Color bgColor;
        Color hoverColor;
        bool isHovered;
        float hoverAnimation;
        
        Button(int x, int y, int w, int h, const std::string& t, Color bg, Color hover) 
            : rect{x, y, w, h}, text(t), bgColor(bg), hoverColor(hover), isHovered(false), hoverAnimation(0.0f) {}
    };
    
    Button playButton;
    Button submitButton;
    Button clearButton;
    Button newGameButton;
    
public:
    SkriblooAIGUI() : window(nullptr), renderer(nullptr), font(nullptr), titleFont(nullptr), smallFont(nullptr),
                      currentState(MENU), isDrawing(false), animationTime(0), pulsePhase(0), confidence(0.0f),
                      playButton(WINDOW_WIDTH/2 - 100, 300, 200, 60, "Start Drawing!", Colors::PRIMARY, Colors::PRIMARY_HOVER),
                      submitButton(500, 270, 120, 50, "Submit", Colors::SUCCESS, Color(34, 217, 114)),
                      clearButton(500, 340, 120, 50, "Clear", Colors::WARNING, Color(255, 178, 31)),
                      newGameButton(WINDOW_WIDTH/2 - 100, 400, 200, 60, "Draw Again", Colors::ACCENT, Color(159, 112, 255)) {
        
        canvas.resize(CANVAS_SIZE, std::vector<bool>(CANVAS_SIZE, false));
        srand(time(nullptr));
        
        // Load the trained CNN model
        model.load("simple_cnn_model.txt");
        std::cout << "CNN model loaded successfully!" << std::endl;
    }
    
    bool initialize() {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
            return false;
        }
        
        if (TTF_Init() == -1) {
            std::cerr << "SDL_ttf could not initialize! SDL_ttf Error: " << TTF_GetError() << std::endl;
            return false;
        }
        
        window = SDL_CreateWindow("SkriblooAI - AI Drawing Guesser", 
                                 SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                 WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
        if (window == nullptr) {
            std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
            return false;
        }
        
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (renderer == nullptr) {
            std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
            return false;
        }
        
        // Load multiple font sizes for better typography
        loadFonts();
        
        return true;
    }
    
    void loadFonts() {
        // Try different font paths for cross-platform compatibility
        std::vector<std::string> fontPaths = {
            "/System/Library/Fonts/SF-Pro-Display-Medium.otf",  // macOS modern
            "/System/Library/Fonts/Helvetica.ttc",             // macOS fallback
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  // Linux
            "C:\\Windows\\Fonts\\segoeui.ttf",                 // Windows modern
            "C:\\Windows\\Fonts\\arial.ttf"                    // Windows fallback
        };
        
        for (const auto& path : fontPaths) {
            titleFont = TTF_OpenFont(path.c_str(), 36);
            font = TTF_OpenFont(path.c_str(), 24);
            smallFont = TTF_OpenFont(path.c_str(), 18);
            
            if (titleFont && font && smallFont) break;
            
            if (titleFont) TTF_CloseFont(titleFont);
            if (font) TTF_CloseFont(font);
            if (smallFont) TTF_CloseFont(smallFont);
            titleFont = font = smallFont = nullptr;
        }
        
        if (!font) {
            std::cout << "Warning: Could not load fonts. Using basic rendering." << std::endl;
        }
    }
    
    void cleanup() {
        if (titleFont) TTF_CloseFont(titleFont);
        if (font) TTF_CloseFont(font);
        if (smallFont) TTF_CloseFont(smallFont);
        if (renderer) SDL_DestroyRenderer(renderer);
        if (window) SDL_DestroyWindow(window);
        TTF_Quit();
        SDL_Quit();
    }
    
    void setRenderColor(const Color& color) {
        SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
    }
    
    void drawText(const std::string& text, int x, int y, TTF_Font* useFont, const Color& color, bool centered = false) {
        if (!useFont) {
            // Fallback rectangle rendering
            setRenderColor(color);
            int width = text.length() * 8;
            if (centered) x -= width / 2;
            SDL_Rect textRect = {x, y, width, 16};
            SDL_RenderDrawRect(renderer, &textRect);
            return;
        }
        
        SDL_Color sdlColor = {color.r, color.g, color.b, color.a};
        SDL_Surface* textSurface = TTF_RenderText_Blended(useFont, text.c_str(), sdlColor);
        if (textSurface) {
            SDL_Texture* textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
            if (textTexture) {
                int textX = centered ? x - textSurface->w / 2 : x;
                SDL_Rect textRect = {textX, y, textSurface->w, textSurface->h};
                SDL_RenderCopy(renderer, textTexture, nullptr, &textRect);
                SDL_DestroyTexture(textTexture);
            }
            SDL_FreeSurface(textSurface);
        }
    }
    
    void drawModernButton(Button& button, int mouseX, int mouseY) {
        // Check if mouse is hovering
        bool wasHovered = button.isHovered;
        button.isHovered = isPointInRect(mouseX, mouseY, button.rect);
        
        // Smooth hover animation
        float targetAnimation = button.isHovered ? 1.0f : 0.0f;
        button.hoverAnimation += (targetAnimation - button.hoverAnimation) * 0.15f;
        
        // Interpolate colors
        Color currentColor = {
            (uint8_t)(button.bgColor.r + (button.hoverColor.r - button.bgColor.r) * button.hoverAnimation),
            (uint8_t)(button.bgColor.g + (button.hoverColor.g - button.bgColor.g) * button.hoverAnimation),
            (uint8_t)(button.bgColor.b + (button.hoverColor.b - button.bgColor.b) * button.hoverAnimation),
            255
        };
        
        // Draw button with rounded corners effect (multiple rectangles)
        int elevation = (int)(4 * button.hoverAnimation);
        SDL_Rect shadowRect = {button.rect.x + 2, button.rect.y + 2 + elevation, button.rect.w, button.rect.h};
        
        // Shadow
        setRenderColor(Color(0, 0, 0, 30));
        SDL_RenderFillRect(renderer, &shadowRect);
        
        // Main button
        SDL_Rect mainRect = {button.rect.x, button.rect.y - elevation, button.rect.w, button.rect.h};
        setRenderColor(currentColor);
        SDL_RenderFillRect(renderer, &mainRect);
        
        // Border highlight
        setRenderColor(Color(255, 255, 255, (uint8_t)(20 + 10 * button.hoverAnimation)));
        SDL_RenderDrawRect(renderer, &mainRect);
        
        // Text
        int textY = mainRect.y + (mainRect.h - 24) / 2;
        drawText(button.text, mainRect.x + mainRect.w / 2, textY, font, Colors::TEXT_PRIMARY, true);
    }
    
    void drawGradientBackground() {
        // Simple gradient effect using horizontal lines
        for (int y = 0; y < WINDOW_HEIGHT; y++) {
            float ratio = (float)y / WINDOW_HEIGHT;
            uint8_t r = (uint8_t)(Colors::BACKGROUND.r + (Colors::SURFACE.r - Colors::BACKGROUND.r) * ratio);
            uint8_t g = (uint8_t)(Colors::BACKGROUND.g + (Colors::SURFACE.g - Colors::BACKGROUND.g) * ratio);
            uint8_t b = (uint8_t)(Colors::BACKGROUND.b + (Colors::SURFACE.b - Colors::BACKGROUND.b) * ratio);
            
            SDL_SetRenderDrawColor(renderer, r, g, b, 255);
            SDL_RenderDrawLine(renderer, 0, y, WINDOW_WIDTH, y);
        }
    }
    
    void drawCanvas() {
        // Canvas shadow
        SDL_Rect shadowRect = {CANVAS_X + 4, CANVAS_Y + 4, CANVAS_SIZE, CANVAS_SIZE};
        setRenderColor(Color(0, 0, 0, 40));
        SDL_RenderFillRect(renderer, &shadowRect);
        
        // Canvas border with gradient effect
        SDL_Rect borderRect = {CANVAS_X - 3, CANVAS_Y - 3, CANVAS_SIZE + 6, CANVAS_SIZE + 6};
        setRenderColor(Colors::CANVAS_BORDER);
        SDL_RenderFillRect(renderer, &borderRect);
        
        // Canvas background
        SDL_Rect canvasRect = {CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE};
        setRenderColor(Colors::CANVAS_BG);
        SDL_RenderFillRect(renderer, &canvasRect);
        
        // Grid pattern (subtle)
        setRenderColor(Color(240, 240, 240));
        for (int i = 28; i < CANVAS_SIZE; i += 28) {
            SDL_RenderDrawLine(renderer, CANVAS_X + i, CANVAS_Y, CANVAS_X + i, CANVAS_Y + CANVAS_SIZE);
            SDL_RenderDrawLine(renderer, CANVAS_X, CANVAS_Y + i, CANVAS_X + CANVAS_SIZE, CANVAS_Y + i);
        }
        
        // Draw the drawing with anti-aliasing effect
        setRenderColor(Color(20, 20, 20));
        for (int y = 0; y < CANVAS_SIZE; y++) {
            for (int x = 0; x < CANVAS_SIZE; x++) {
                if (canvas[y][x]) {
                    // Draw main pixel
                    SDL_RenderDrawPoint(renderer, CANVAS_X + x, CANVAS_Y + y);
                    
                    // Add slight anti-aliasing
                    setRenderColor(Color(100, 100, 100, 100));
                    if (x > 0 && !canvas[y][x-1]) SDL_RenderDrawPoint(renderer, CANVAS_X + x - 1, CANVAS_Y + y);
                    if (x < CANVAS_SIZE-1 && !canvas[y][x+1]) SDL_RenderDrawPoint(renderer, CANVAS_X + x + 1, CANVAS_Y + y);
                    if (y > 0 && !canvas[y-1][x]) SDL_RenderDrawPoint(renderer, CANVAS_X + x, CANVAS_Y + y - 1);
                    if (y < CANVAS_SIZE-1 && !canvas[y+1][x]) SDL_RenderDrawPoint(renderer, CANVAS_X + x, CANVAS_Y + y + 1);
                    setRenderColor(Color(20, 20, 20));
                }
            }
        }
    }
    
    void drawProgressBar(float progress, int x, int y, int width, int height) {
        // Background
        SDL_Rect bgRect = {x, y, width, height};
        setRenderColor(Colors::SURFACE);
        SDL_RenderFillRect(renderer, &bgRect);
        
        // Progress
        SDL_Rect progressRect = {x, y, (int)(width * progress), height};
        setRenderColor(Colors::PRIMARY);
        SDL_RenderFillRect(renderer, &progressRect);
        
        // Border
        setRenderColor(Colors::CANVAS_BORDER);
        SDL_RenderDrawRect(renderer, &bgRect);
    }
    
    void drawPulsingCircle(int x, int y, int radius, const Color& color) {
        float pulse = sin(animationTime * 4.0f) * 0.3f + 0.7f;
        int currentRadius = (int)(radius * pulse);
        
        // Draw filled circle (approximate with multiple rectangles)
        for (int dy = -currentRadius; dy <= currentRadius; dy++) {
            for (int dx = -currentRadius; dx <= currentRadius; dx++) {
                if (dx*dx + dy*dy <= currentRadius*currentRadius) {
                    uint8_t alpha = (uint8_t)(color.a * (1.0f - sqrt(dx*dx + dy*dy) / currentRadius) * pulse);
                    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, alpha);
                    SDL_RenderDrawPoint(renderer, x + dx, y + dy);
                }
            }
        }
    }
    
    void handleDrawing(int mouseX, int mouseY) {
        int canvasX = mouseX - CANVAS_X;
        int canvasY = mouseY - CANVAS_Y;
        
        if (canvasX >= 0 && canvasX < CANVAS_SIZE && canvasY >= 0 && canvasY < CANVAS_SIZE) {
            // Draw with smooth brush
            int brushSize = 4;
            for (int dy = -brushSize; dy <= brushSize; dy++) {
                for (int dx = -brushSize; dx <= brushSize; dx++) {
                    int x = canvasX + dx;
                    int y = canvasY + dy;
                    float distance = sqrt(dx*dx + dy*dy);
                    if (x >= 0 && x < CANVAS_SIZE && y >= 0 && y < CANVAS_SIZE && distance <= brushSize) {
                        canvas[y][x] = true;
                    }
                }
            }
        }
    }
    
    void clearCanvas() {
        for (int y = 0; y < CANVAS_SIZE; y++) {
            for (int x = 0; x < CANVAS_SIZE; x++) {
                canvas[y][x] = false;
            }
        }
    }
    
    void startDrawing() {
        clearCanvas();
        currentState = DRAWING;
        prediction = "";
        secondPrediction = "";
        thirdPrediction = "";
        confidence = 0.0f;
        secondConfidence = 0.0f;
        thirdConfidence = 0.0f;
    }
    
    void submitDrawing() {
        std::vector<std::vector<uint8_t>> image28x28(28, std::vector<uint8_t>(28, 0));
        
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int blackPixels = 0;
                for (int dy = 0; dy < 10; dy++) {
                    for (int dx = 0; dx < 10; dx++) {
                        int origX = x * 10 + dx;
                        int origY = y * 10 + dy;
                        if (origX < CANVAS_SIZE && origY < CANVAS_SIZE && canvas[origY][origX]) {
                            blackPixels++;
                        }
                    }
                }
                image28x28[y][x] = (blackPixels > 0) ? 255 : 0;
            }
        }
        
        saveImage28x28("current_drawing.png", image28x28);
        analyzeDrawing("current_drawing.png");
        currentState = RESULT;
    }
    
    void saveImage28x28(const std::string& filename, const std::vector<std::vector<uint8_t>>& image) {
        cv::Mat img(28, 28, CV_8UC1);
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                img.at<uint8_t>(y, x) = image[y][x];
            }
        }
        cv::imwrite(filename, img);
    }
    
    void analyzeDrawing(const std::string& filename) {
        try {
            Eigen::Tensor<double, 3> input_tensor = model.convert_images(filename);
            auto [probabilities, loss] = model.forward_pass(input_tensor, 0);
            
            // Create a vector of probability-index pairs for sorting
            std::vector<std::pair<double, int>> prob_pairs;
            for (int i = 0; i < probabilities.size(); ++i) {
                prob_pairs.push_back({probabilities(i), i});
            }
            
            // Sort by probability in descending order
            std::sort(prob_pairs.begin(), prob_pairs.end(), std::greater<>());
            
            // Get top 3 predictions
            if (prob_pairs.size() >= 1) {
                confidence = prob_pairs[0].first;
                int predicted_class = prob_pairs[0].second;
                if (predicted_class >= 0 && predicted_class < (int)categories.size()) {
                    prediction = categories[predicted_class];
                }
            }
            
            if (prob_pairs.size() >= 2) {
                secondConfidence = prob_pairs[1].first;
                int second_class = prob_pairs[1].second;
                if (second_class >= 0 && second_class < (int)categories.size()) {
                    secondPrediction = categories[second_class];
                }
            }
            
            if (prob_pairs.size() >= 3) {
                thirdConfidence = prob_pairs[2].first;
                int third_class = prob_pairs[2].second;
                if (third_class >= 0 && third_class < (int)categories.size()) {
                    thirdPrediction = categories[third_class];
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error in CNN prediction: " << e.what() << std::endl;
            prediction = "error";
        }
    }
    
    bool isPointInRect(int x, int y, const SDL_Rect& rect) {
        return (x >= rect.x && x < rect.x + rect.w && y >= rect.y && y < rect.y + rect.h);
    }
    
    void handleEvents() {
        SDL_Event e;
        int mouseX, mouseY;
        SDL_GetMouseState(&mouseX, &mouseY);
        
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                exit(0);
            }
            
            if (e.type == SDL_MOUSEBUTTONDOWN) {
                if (currentState == MENU && isPointInRect(mouseX, mouseY, playButton.rect)) {
                    startDrawing();
                }
                else if (currentState == DRAWING) {
                    if (isPointInRect(mouseX, mouseY, submitButton.rect)) {
                        submitDrawing();
                    }
                    else if (isPointInRect(mouseX, mouseY, clearButton.rect)) {
                        clearCanvas();
                    }
                    else if (mouseX >= CANVAS_X && mouseX < CANVAS_X + CANVAS_SIZE &&
                             mouseY >= CANVAS_Y && mouseY < CANVAS_Y + CANVAS_SIZE) {
                        isDrawing = true;
                        handleDrawing(mouseX, mouseY);
                    }
                }
                else if (currentState == RESULT && isPointInRect(mouseX, mouseY, newGameButton.rect)) {
                    currentState = MENU;
                }
            }
            
            if (e.type == SDL_MOUSEBUTTONUP) {
                isDrawing = false;
            }
            
            if (e.type == SDL_MOUSEMOTION && isDrawing) {
                handleDrawing(mouseX, mouseY);
            }
        }
    }
    
    void render() {
        // Update animation
        animationTime += 0.016f;
        
        // Get mouse position for button hover effects
        int mouseX, mouseY;
        SDL_GetMouseState(&mouseX, &mouseY);
        
        drawGradientBackground();
        
        if (currentState == MENU) {
            // Title with AI brain symbol
            drawText("Skribloo", WINDOW_WIDTH/2, 50, titleFont, Colors::TEXT_PRIMARY, true);
            
            // Subtitle
            drawText("Can a Neural Network learn doodling?", WINDOW_WIDTH/2, 120, font, Colors::TEXT_SECONDARY, true);
            
            // Animated decorative circles
            drawPulsingCircle(220, 380, 25, Colors::PRIMARY);
            drawPulsingCircle(220, 280, 25, Colors::ACCENT);
            drawPulsingCircle(WINDOW_WIDTH - 220, 380, 20, Colors::ACCENT);
            drawPulsingCircle(WINDOW_WIDTH - 220, 280, 20, Colors::PRIMARY);            
            drawModernButton(playButton, mouseX, mouseY);
            
            // Footer with categories
            drawText("Categoreis: airplane, apple, bicycle, book, car, cat, chair, clock, cloud, cup", 
                    WINDOW_WIDTH/2, WINDOW_HEIGHT - 50, smallFont, Colors::TEXT_SECONDARY, true);
        }
        else if (currentState == DRAWING) {
            // Header
            drawText("Draw Something!", WINDOW_WIDTH/2, 40, titleFont, Colors::PRIMARY, true);            
            drawCanvas();
            
            // Side panel with instructions
            drawText("Instructions:", 480, 120, font, Colors::TEXT_PRIMARY);
            drawText("Draw anything in the canvas", 480, 150, smallFont, Colors::TEXT_SECONDARY);
            drawText("Use your mouse to draw", 480, 175, smallFont, Colors::TEXT_SECONDARY);
            drawText("Click submit when finished", 480, 200, smallFont, Colors::TEXT_SECONDARY);
            drawText("AI will try to guess it!", 480, 225, smallFont, Colors::ACCENT);
            
            drawModernButton(submitButton, mouseX, mouseY);
            drawModernButton(clearButton, mouseX, mouseY);
        }
        else if (currentState == RESULT) {
            // Results with styling
            drawText("Neural Network's Analysis", WINDOW_WIDTH/2, 60, titleFont, Colors::TEXT_PRIMARY, true);            
            drawText("Here's what I think you drew:", WINDOW_WIDTH/2, 110, font, Colors::TEXT_SECONDARY, true);
            
            // Top 3 AI predictions
            int yOffset = 150;
            
            if (!prediction.empty() && prediction != "error") {
                // First prediction (most confident)
                drawText("1st Guess: " + prediction, WINDOW_WIDTH/2, yOffset, font, Colors::SUCCESS, true);
                drawProgressBar(confidence, WINDOW_WIDTH/2 - 100, yOffset + 25, 200, 15);
                std::string confidence1Text = std::to_string((int)(confidence * 100)) + "% confident";
                drawText(confidence1Text, WINDOW_WIDTH/2, yOffset + 45, smallFont, Colors::TEXT_PRIMARY, true);
                yOffset += 80;
                
                // Second prediction
                if (!secondPrediction.empty()) {
                    drawText("2nd Guess: " + secondPrediction, WINDOW_WIDTH/2, yOffset, font, Colors::PRIMARY, true);
                    drawProgressBar(secondConfidence, WINDOW_WIDTH/2 - 100, yOffset + 25, 200, 15);
                    std::string confidence2Text = std::to_string((int)(secondConfidence * 100)) + "% confident";
                    drawText(confidence2Text, WINDOW_WIDTH/2, yOffset + 45, smallFont, Colors::TEXT_SECONDARY, true);
                    yOffset += 70;
                }
                
                // Third prediction
                if (!thirdPrediction.empty()) {
                    drawText("3rd Guess: " + thirdPrediction, WINDOW_WIDTH/2, yOffset, font, Colors::TEXT_SECONDARY, true);
                    drawProgressBar(thirdConfidence, WINDOW_WIDTH/2 - 100, yOffset + 25, 200, 15);
                    std::string confidence3Text = std::to_string((int)(thirdConfidence * 100)) + "% confident";
                    drawText(confidence3Text, WINDOW_WIDTH/2, yOffset + 45, smallFont, Colors::TEXT_SECONDARY, true);
                    yOffset += 70;
                }
            } else {
                drawText("Hmm, I'm not sure what this is!", WINDOW_WIDTH/2, yOffset, font, Colors::WARNING, true);
                drawText("Try drawing something else!", WINDOW_WIDTH/2, yOffset + 30, font, Colors::TEXT_SECONDARY, true);
            }
            
            drawModernButton(newGameButton, mouseX, mouseY);
        }
        
        SDL_RenderPresent(renderer);
    }
    
    void run() {
        while (true) {
            handleEvents();
            render();
            SDL_Delay(16);  // ~60 FPS
        }
    }
};

int main() {
    SkriblooAIGUI game;
    
    if (!game.initialize()) {
        std::cerr << "Failed to initialize!" << std::endl;
        return -1;
    }
    
    game.run();
    game.cleanup();
    
    return 0;
}
