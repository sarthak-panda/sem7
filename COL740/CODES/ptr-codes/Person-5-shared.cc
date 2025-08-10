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

    // Deleted copy operations (cannot copy shared resource this way)
    Person(const Person&) = delete;
    Person& operator=(const Person&) = delete;

    // Destructor
    ~Person() {
        if (name) {
            std::cout << "Destructor called for " << name.get() << std::endl;
        } else {
            std::cout << "Destructor called for unnamed Person" << std::endl;
        }
    }

    void display() const {
        std::cout << "Name: " << (name ? name.get() : "None") << ", Age: " << age << std::endl;
    }
};

// Function that shares ownership
void sharePerson(std::shared_ptr<Person> p) {
    std::cout << "Inside sharePerson function. Use count: " << p.use_count() << std::endl;
    p->display();
}

int main() {
    // Create a shared_ptr
    std::shared_ptr<Person> p1 = std::make_shared<Person>("Alice", 30);
    std::cout << "p1 use count: " << p1.use_count() << std::endl;

    {
        // Create another shared_ptr sharing the same Person
        std::shared_ptr<Person> p2 = p1;
        std::cout << "p1 use count after p2 created: " << p1.use_count() << std::endl;

        sharePerson(p2);  // Pass shared_ptr by value (increases count)
        std::cout << "p1 use count after sharePerson: " << p1.use_count() << std::endl;
    } // p2 goes out of scope here

    std::cout << "p1 use count after p2 destroyed: " << p1.use_count() << std::endl;

    // Reset p1, Person will be destroyed automatically
    std::shared_ptr<Person> px = p1;

    p1.reset();
    std::cout << "Use count after reset = " << px.use_count() << std::endl << std::endl; 
    std::cout << "p1 reset called. If this was the last owner, Person is destroyed." << std::endl;

    return 0;
}

