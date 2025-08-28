#include <iostream>
#include <memory>

// -------------------- Abstract Products --------------------
class Shape {
public:
    virtual void draw() const = 0;
    virtual ~Shape() = default;
};

class Color {
public:
    virtual void fill() const = 0;
    virtual ~Color() = default;
};

// -------------------- Concrete Products --------------------
class Circle : public Shape {
public:
    void draw() const override { std::cout << "Drawing Circle\n"; }
};
class Square : public Shape {
public:
    void draw() const override { std::cout << "Drawing Square\n"; }
};

class Red : public Color {
public:
    void fill() const override { std::cout << "Filling with Red\n"; }
};
class Blue : public Color {
public:
    void fill() const override { std::cout << "Filling with Blue\n"; }
};

// -------------------- Abstract Factory --------------------
class AbstractFactory {
public:
    virtual std::unique_ptr<Shape> createShape() const = 0;
    virtual std::unique_ptr<Color> createColor() const = 0;
    virtual ~AbstractFactory() = default;
};

// -------------------- Concrete Factories --------------------
class RoundFactory : public AbstractFactory {
public:
    std::unique_ptr<Shape> createShape() const override {
        return std::make_unique<Circle>();
    }
    std::unique_ptr<Color> createColor() const override {
        return std::make_unique<Red>();
    }
};

class NormalFactory : public AbstractFactory {
public:
    std::unique_ptr<Shape> createShape() const override {
        return std::make_unique<Square>();
    }
    std::unique_ptr<Color> createColor() const override {
        return std::make_unique<Blue>();
    }
};

// -------------------- Client --------------------
int main() {
    // Choose a factory family
    std::unique_ptr<AbstractFactory> factory = std::make_unique<RoundFactory>();

    auto shape = factory->createShape();
    auto color = factory->createColor();

    shape->draw();
    color->fill();

    return 0;
}

