#include <iostream>
#include <cstring>

class Person {
private:
    char* name;   // dynamically allocated memory for name
    int age;

public:
    // Constructor
    Person(const char* n, int a) : age(a) {
        name = new char[strlen(n) + 1];  // allocate memory
        strcpy(name, n);                 // copy the string
        std::cout << "Constructor called for " << name << std::endl;
    }

    // Copy constructor (deep copy)
    Person(const Person &other) : age(other.age) {
        name = new char[strlen(other.name) + 1];  // allocate new memory
        strcpy(name, other.name);                 // copy contents
        std::cout << "Copy constructor called for " << name << std::endl;
    }

    // Destructor
    ~Person() {
        std::cout << "Destructor called for " << name << " " << age <<  std::endl;
        delete[] name;  // free allocated memory
    }

    // Display function
    void display() const {
        std::cout << "Name: " << name << ", Age: " << age << std::endl;
    }
    void setAge(int age) {
        this->age = age;
    }
};

int main() {
    Person p1("Alice", 30);
    p1.display();

    // Create a copy using copy constructor
    Person p2 = p1;  // deep copy
    p2.setAge(31);
    p2.display();

    // Another copy
    Person p3(p1);
    p3.setAge(32);
    p3.display();

    return 0;
}
