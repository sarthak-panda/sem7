#include <iostream>
#include <mutex>

class Singleton {
private:
    // Private constructor: prevents direct instantiation
    Singleton() {
        std::cout << "Singleton instance created.\n";
    }

    // Deleted copy constructor and assignment operator
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

public:
    // Static accessor: returns reference to the unique instance
    static Singleton& getInstance() {
        // C++11 guarantees thread-safe initialization of static local variables
        static Singleton instance;
        return instance;
    }

    // Example method
    void doSomething() {
        std::cout << "Doing something in the singleton instance.\n";
    }
};

int main() {
    // Access the singleton instance
    Singleton& s1 = Singleton::getInstance();
    s1.doSomething();

    // Any subsequent calls return the same instance
    Singleton& s2 = Singleton::getInstance();

    if (&s1 == &s2) {
        std::cout << "Both references point to the same instance.\n";
    }

    return 0;
}

