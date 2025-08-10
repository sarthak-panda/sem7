#include <iostream>
#include <string>

class Person {
private:
    std::string name;
    int age;

public:
    // Default constructor
    Person(const std::string &n, int a) : name(n), age(a) {
        std::cout << "Constructor called for " << name << std::endl;
    }

    // Copy constructor
    Person(const Person &other) {
        name = other.name;
        age = other.age;
        std::cout << "Copy constructor called for " << name << std::endl;
    }

    // Display function
    void display() const {
        std::cout << "Name: " << name << ", Age: " << age << std::endl;
    }
};

int main() {
    // Create an object
    Person p1("Alice", 30);
    p1.display();

    // Create a copy using the copy constructor
    Person p2 = p1;  // Copy constructor is invoked
    p2.display();

    // Another way: explicitly calling copy constructor
    Person p3(p1);
    p3.display();

    return 0;
}
