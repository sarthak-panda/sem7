#include <iostream>
#include <memory>
#include <vector>
#include <string>

// CRTP for type-safe cloning
template <typename Derived>
class Cloneable {
public:
    std::shared_ptr<Derived> clone() const {
        return std::make_shared<Derived>(static_cast<const Derived&>(*this));
    }
};

// Base class with virtual inheritance and shared_from_this
class Animal : public std::enable_shared_from_this<Animal> {
public:
    virtual void speak() const = 0;

    virtual std::shared_ptr<Animal> getSelf() {
        return shared_from_this();
    }

    virtual ~Animal() = default;
};

// Diamond hierarchy
class Mammal : virtual public Animal {};
class Bird : virtual public Animal {};

// Derived class: Bat
class Bat : public Mammal, public Bird, public Cloneable<Bat> {
    std::string name;

public:
    Bat(const std::string& n) : name(n) {
        std::cout << "Bat constructed: " << name << std::endl;
    }

    Bat(const Bat& other) : name(other.name + "_copy") {
        std::cout << "Bat copied: " << name << std::endl;
    }

    void speak() const override {
        std::cout << name << " says squeak!\n";
    }

    // Correct: same return type as in Animal
    std::shared_ptr<Animal> getSelf() override {
        return shared_from_this(); // returns shared_ptr<Animal>
    }

    // Extra: type-safe version if needed
    std::shared_ptr<Bat> getBatSelf() {
        return std::dynamic_pointer_cast<Bat>(shared_from_this());
    }
};

int main() {
    std::shared_ptr<Bat> bat = std::make_shared<Bat>("Bruce");

    std::vector<std::shared_ptr<Animal>> zoo;

    // All methods return shared_ptr<Animal> safely
    zoo.push_back(bat->getSelf());
    zoo.push_back(bat->clone()); // Returns shared_ptr<Bat>, implicitly converted
    zoo.push_back(std::dynamic_pointer_cast<Animal>(bat));

    for (const auto& animal : zoo) {
        std::cout << std::endl;
        animal->speak();
    }
}

