#include <iostream>
#include <map>
#include <memory>

// -------- Flyweight (shared object) --------
class Character {
    char symbol; // intrinsic state
public:
    explicit Character(char c) : symbol(c) {}
    void display(int fontSize) const { // extrinsic state passed in
        std::cout << "Char '" << symbol << "' in size " << fontSize << "\n";
    }
};

// -------- Flyweight Factory --------
class CharacterFactory {
    std::map<char, std::shared_ptr<Character>> pool;
public:
    std::shared_ptr<Character> get(char c) {
        if (!pool.count(c)) pool[c] = std::make_shared<Character>(c);
        return pool[c];
    }
};

// -------- Client --------
int main() {
    CharacterFactory factory;

    // get() reuses flyweights for same symbol
    auto a1 = factory.get('a');
    auto a2 = factory.get('a');
    auto b  = factory.get('b');

    a1->display(12);
    a2->display(16); // same flyweight object, different extrinsic state
    b->display(14);

    if (a1 == a2) std::cout << "a1 and a2 are the same instance!\n";
}

