#include <iostream>
#include <memory>

// -------- Component --------
class Component {
public:
    virtual void operation() const = 0;
    virtual ~Component() = default;
};

// -------- Concrete Component --------
class ConcreteComponent : public Component {
public:
    void operation() const override {
        std::cout << "ConcreteComponent";
    }
};

// -------- Base Decorator --------
class Decorator : public Component {
protected:
    std::unique_ptr<Component> wrappee;
public:
    Decorator(std::unique_ptr<Component> c) : wrappee(std::move(c)) {}
};

// -------- Concrete Decorators --------
class DecoratorA : public Decorator {
public:
    using Decorator::Decorator; // inherit constructor
    void operation() const override {
        wrappee->operation();
        std::cout << " + DecoratorA";
    }
};

class DecoratorB : public Decorator {
public:
    using Decorator::Decorator;
    void operation() const override {
        wrappee->operation();
        std::cout << " + DecoratorB";
    }
};

// -------- Client --------
int main() {
    std::unique_ptr<Component> obj = std::make_unique<ConcreteComponent>();
    obj = std::make_unique<DecoratorA>(std::move(obj));
    obj = std::make_unique<DecoratorB>(std::move(obj));

    obj->operation(); // Output: ConcreteComponent + DecoratorA + DecoratorB
    std::cout << "\n";
}

