#include <iostream>
#include <memory>

class Person : public std::enable_shared_from_this<Person> {
public:
    std::string name;

    Person(std::string n) : name(std::move(n)) {}
    
    std::shared_ptr<Person> getPtr() {
        return shared_from_this();
    }

    void greet() {
        std::cout << "Hello, I am " << name << std::endl;
    }
};

int main() {
    auto p = std::make_shared<Person>("Alice");
    auto p2 = p->getPtr(); // creates another shared_ptr to same object
    
    std::cout << "Use count: " << p.use_count() << std::endl;
    p2->greet();
    return 0;
}

