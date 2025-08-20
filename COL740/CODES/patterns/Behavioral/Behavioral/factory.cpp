#include <iostream>
#include <memory>   // for std::unique_ptr
#include <string>

// -------------------- Product Interface --------------------
class Shape {
public:
    virtual void draw() const = 0;  // pure virtual
    virtual ~Shape() = default;
};

// -------------------- Concrete Products --------------------
class Circle : public Shape {
public:
    void draw() const override {
        std::cout << "Drawing a Circle\n";
    }
};

class Square : public Shape {
public:
    void draw() const override {
        std::cout << "Drawing a Square\n";
    }
};

// -------------------- Factory --------------------
class ShapeFactory {
public:
    // Factory Method
    static std::unique_ptr<Shape> createShape(const std::string& type) {
        if (type == "circle") {
            return std::make_unique<Circle>();
        } else if (type == "square") {
            return std::make_unique<Square>();
        } else {
            return nullptr;
        }
    }
};

// -------------------- Client Code --------------------
int main() {
    auto shape1 = ShapeFactory::createShape("circle");
    auto shape2 = ShapeFactory::createShape("square");

    if (shape1) shape1->draw();
    if (shape2) shape2->draw();

    return 0;
}

