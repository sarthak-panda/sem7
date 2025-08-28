#include <iostream>
#include <memory>

// -------------------- Implementor Interface --------------------
class DrawingAPI {
public:
    virtual void drawCircle(double x, double y, double radius) = 0;
    virtual ~DrawingAPI() = default;
};

// -------------------- Concrete Implementors --------------------
class DrawingAPI1 : public DrawingAPI {
public:
    void drawCircle(double x, double y, double radius) override {
        std::cout << "API1.circle at (" << x << "," << y 
                  << ") radius " << radius << "\n";
    }
};

class DrawingAPI2 : public DrawingAPI {
public:
    void drawCircle(double x, double y, double radius) override {
        std::cout << "API2.circle at (" << x << "," << y 
                  << ") radius " << radius << "\n";
    }
};

// -------------------- Abstraction --------------------
class Shape {
protected:
    std::shared_ptr<DrawingAPI> drawingAPI;
public:
    Shape(std::shared_ptr<DrawingAPI> api) : drawingAPI(api) {}
    virtual void draw() = 0;
    virtual ~Shape() = default;
};

// -------------------- Refined Abstraction --------------------
class CircleShape : public Shape {
    double x, y, radius;
public:
    CircleShape(double x, double y, double r, std::shared_ptr<DrawingAPI> api)
        : Shape(api), x(x), y(y), radius(r) {}

    void draw() override {
        drawingAPI->drawCircle(x, y, radius);
    }
};

// -------------------- Client --------------------
int main() {
    auto api1 = std::make_shared<DrawingAPI1>();
    auto api2 = std::make_shared<DrawingAPI2>();

    CircleShape c1(1, 2, 3, api1);
    CircleShape c2(5, 7, 11, api2);

    c1.draw();  // uses API1
    c2.draw();  // uses API2

    return 0;
}

