#include <iostream>
#include <memory>

class B;  // Forward declaration

class A {
public:
	std::shared_ptr<B> b_ptr;  // Use weak_ptr instead of shared_ptr

    ~A() {
        std::cout << "Destructor of A called" << std::endl;
    }
};

class B {
public:
    std::shared_ptr<A> a_ptr;  // B owns A

    ~B() {
        std::cout << "Destructor of B called" << std::endl;
    }
};

int main() {
    std::shared_ptr<A> a = std::make_shared<A>();
    std::shared_ptr<B> b = std::make_shared<B>();

    a->b_ptr = b;  // A has a refererence to B
    b->a_ptr = a;  // B has a reference to A 

    std::cout << "Nothing will happen" << std::endl;
    return 0;
}
