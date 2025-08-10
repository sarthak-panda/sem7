#include <iostream>
#include <memory>

struct Person {
    std::string name;
    int age;
};

int main() {
    auto person = std::make_shared<Person>(Person{"Alice", 30});
    
    // Aliasing constructor: shared_ptr owns `person`, but points to `person->name`
    std::shared_ptr<std::string> namePtr(person, &person->name);

    std::cout << "Name via aliasing: " << *namePtr << std::endl;
    std::cout << "Count = " << person.use_count() << std::endl;
    return 0;
}

