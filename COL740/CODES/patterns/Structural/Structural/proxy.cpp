#include <iostream>
#include <memory>
#include <string>

// -------- Subject --------
class Image {
public:
    virtual void display() = 0;
    virtual ~Image() = default;
};

// -------- Real Subject (expensive to create) --------
class RealImage : public Image {
    std::string file;
public:
    explicit RealImage(std::string f) : file(std::move(f)) {
        std::cout << "Loading image from disk: " << file << "\n";
    }
    void display() override {
        std::cout << "Displaying: " << file << "\n";
    }
};

// -------- Proxy --------
class ImageProxy : public Image {
    std::string file;
    std::unique_ptr<RealImage> real;  // created on demand
public:
    explicit ImageProxy(std::string f) : file(std::move(f)) {}
    void display() override {
        if (!real) real = std::make_unique<RealImage>(file); // lazy init
        std::cout << "(Proxy) ";
        real->display();
    }
};

// -------- Client --------
int main() {
    Image* img = new ImageProxy("diagram.png");
    std::cout << "App started; image not loaded yet.\n";

    // Only when needed:
    img->display();   // Loads then displays
    img->display();   // Reuses already-loaded RealImage

    delete img;
}

