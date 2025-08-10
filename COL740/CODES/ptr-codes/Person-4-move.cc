#include <iostream>
#include <cstring>
#include <memory>

class Person {
private:
    std::unique_ptr<char[]> name;
    int age;

public:
    // Constructor
    Person(const char* n, int a) : age(a) {
        size_t len = std::strlen(n) + 1;
        name = std::make_unique<char[]>(len);
        std::strcpy(name.get(), n);
        std::cout << "Constructor called for " << name.get() << std::endl;
    }

    // Move constructor
    Person(Person&& other) noexcept : name(std::move(other.name)), age(other.age) {
        other.age = 0;
        std::cout << "Move constructor called." << std::endl;
    }

    // Move assignment operator
    Person& operator=(Person&& other) noexcept {
        if (this != &other) {
            name = std::move(other.name);
            age = other.age;
            other.age = 0;
        }
        std::cout << "Move assignment operator called." << std::endl;
        return *this;
    }

    // Deleted copy operations (unique_ptr cannot be copied)
    Person(const Person&) = delete;
    Person& operator=(const Person&) = delete;

    ~Person() {
        if (name) {
            std::cout << "Destructor called for " << name.get() << " " << age << std::endl;
        } else {
            std::cout << "Destructor called for unnamed Person" << " " << age << std::endl;
        }
    }

    void setAge (int a) {
        age = a;
    }

    void display() const {
        std::cout << "Name: " << (name ? name.get() : "None") << ", Age: " << age << std::endl;
    }
};

// Function that takes ownership using move semantics
void processPerson(std::unique_ptr<Person> p) {
    std::cout << "Inside processPerson function:" << std::endl;
    p->display();
}

int main() {
    std::unique_ptr<Person> p1 = std::make_unique<Person>("Alice", 31);
    p1->display();

    // Transfer ownership to function using move semantics
    processPerson(std::move(p1));

    std::cout << "I am at the function return point" << std::endl << std::endl; 

    // After move, p1 is now nullptr
    if (!p1) {
        std::cout << "p1 is now nullptr after move." << std::endl;
    }

    // Create another Person and move into a new pointer
    std::unique_ptr<Person> p2 = std::make_unique<Person>("Bob", 32);
    Person p3 = std::move(*p2);
    p3.setAge(33);

    if (!p2) {
        std::cout << "**** p2 is now nullptr after moving to p3." << std::endl;
    }

    p3.display();

    return 0;
}

