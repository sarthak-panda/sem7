#include <iostream>
#include <cstring>
#include <memory>  // for smart pointers

class Person {
private:
    std::unique_ptr<char[]> name;  // smart pointer for dynamic memory
    int age;

public:
    // Constructor
    Person(const char* n, int a) : age(a) {
        size_t len = std::strlen(n) + 1;
        name = std::make_unique<char[]>(len);  // allocate memory
        std::strcpy(name.get(), n);            // copy string
        std::cout << "Constructor called for " << name.get() << std::endl;
    }

    // Copy constructor (deep copy for unique_ptr)
    Person(const Person& other) : age(other.age) {
        size_t len = std::strlen(other.name.get()) + 1;
        name = std::make_unique<char[]>(len);
        std::strcpy(name.get(), other.name.get());
        std::cout << "Copy constructor called for " << name.get() << std::endl;
    }

    // Copy assignment operator (deep copy)
    Person& operator=(const Person& other) {
        if (this != &other) {
            age = other.age;
            size_t len = std::strlen(other.name.get()) + 1;
            name = std::make_unique<char[]>(len);
            std::strcpy(name.get(), other.name.get());
        }
        std::cout << "Assignment constructor called for " << name.get() << std::endl;
        return *this;
    }

    // Destructor (no need for delete, unique_ptr cleans up)
    ~Person() {
        std::cout << "Destructor called for " << name.get() << std::endl;
    }

    // Display function
    void display() const {
        std::cout << "Name: " << name.get() << ", Age: " << age << std::endl;
    }
};

int main() {
    Person p1("Alice", 30);
    p1.display();

    // Create a copy using copy constructor
    Person p2 = p1;
    p2.display();

    // Another copy using assignment
    Person p3("Temp", 0);
    p3 = p1;
    p3.display();

    return 0;
}

