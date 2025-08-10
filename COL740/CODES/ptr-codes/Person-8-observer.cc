#include <iostream>
#include <memory>

class Person {
public:
    std::string name;
    Person(std::string n) : name(std::move(n)) {}
    void greet() const { std::cout << "Hello, I am " << name << std::endl; }
};

class PersonObserver {
    std::weak_ptr<Person> person; // weak_ptr prevents dangling access
public:
    PersonObserver(std::weak_ptr<Person> p) : person(std::move(p)) {}

    void observe() {
        if (auto sp = person.lock()) { // check if object is still alive
            sp->greet();
        } else {
            std::cout << "Person no longer exists!" << std::endl;
        }
    }
};

int main() {
    auto owner = std::make_shared<Person>("Alice");
    PersonObserver observer(owner);

    std::cout << "Use count in Line 29 = " << owner.use_count() << std::endl; 
    observer.observe(); // Alice is alive

    std::cout << "Use count in Line 32 = " << owner.use_count() << std::endl; 
    owner.reset();      // destroy Alice

    std::cout << "Use count in Line 35 = " << owner.use_count() << std::endl; 
    observer.observe(); // Safe: Person no longer exists!
}
