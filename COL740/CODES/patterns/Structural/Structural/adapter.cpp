#include <iostream>
#include <memory>

// -------------------- Target Interface --------------------
class Printer {
public:
    virtual void print(const std::string& text) = 0;
    virtual ~Printer() = default;
};

// -------------------- Legacy Class (incompatible) --------------------
class OldPrinter {
public:
    void oldPrint(const std::string& msg) {
        std::cout << "[OldPrinter] " << msg << "\n";
    }
};

// -------------------- Adapter (Wrapper) --------------------
class PrinterAdapter : public Printer {
    std::unique_ptr<OldPrinter> adaptee;
public:
    PrinterAdapter() : adaptee(std::make_unique<OldPrinter>()) {}
    
    void print(const std::string& text) override {
        // Translate call to legacy method
        adaptee->oldPrint(text);
    }
};

// -------------------- Client --------------------
int main() {
    std::unique_ptr<Printer> printer = std::make_unique<PrinterAdapter>();
    printer->print("Hello via Adapter!");
    return 0;
}

