#include <iostream>
#include <memory>

// -------------------- Prototype Interface --------------------
class Shape {
public:
    virtual std::unique_ptr<Shape> clone() const = 0;
    virtual void draw() const = 0;
    virtual ~Shape() = default;
};

// -------------------- Concrete Prototypes --------------------
class Circle : public Shape {
public:
    std::unique_ptr<Shape> clone() const override {
        return std::make_unique<Circle>(*this);  // copy constructor
    }
    void draw() const override {
        std::cout << "Drawing a Circle\n";
    }
};

class Square : public Shape {
public:
    std::unique_ptr<Shape> clone() const override {
        return std::make_unique<Square>(*this);  // copy constructor
    }
    void draw() const override {
        std::cout << "Drawing a Square\n";
    }
};

// -------------------- Client --------------------
int main() {
    std::unique_ptr<Shape> c1 = std::make_unique<Circle>();
    std::unique_ptr<Shape> c2 = c1->clone();  // copy instead of new
    c2->draw();

    std::unique_ptr<Shape> s1 = std::make_unique<Square>();
    std::unique_ptr<Shape> s2 = s1->clone();
    s2->draw();

    return 0;
}

