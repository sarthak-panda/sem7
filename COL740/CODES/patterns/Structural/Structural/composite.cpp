#include <iostream>
#include <memory>
#include <string>
#include <vector>

// ---------------- Component ----------------
class Component {
public:
    virtual void operation() const = 0;
    virtual ~Component() = default;
};

// ---------------- Leaf ----------------
class Leaf : public Component {
    std::string name;
public:
    Leaf(std::string n) : name(std::move(n)) {}
    void operation() const override {
        std::cout << "Leaf: " << name << "\n";
    }
};

// ---------------- Composite ----------------
class Composite : public Component {
    std::string name;
    std::vector<std::unique_ptr<Component>> children;
public:
    Composite(std::string n) : name(std::move(n)) {}
    void add(std::unique_ptr<Component> c) { children.push_back(std::move(c)); }
    void operation() const override {
        std::cout << "Composite: " << name << "\n";
        for (const auto& child : children) child->operation();
    }
};

// ---------------- Client ----------------
int main() {
    auto root = std::make_unique<Composite>("root");
    root->add(std::make_unique<Leaf>("A"));
    root->add(std::make_unique<Leaf>("B"));

    auto sub = std::make_unique<Composite>("subtree");
    sub->add(std::make_unique<Leaf>("C"));
    root->add(std::move(sub));

    root->operation();
}

