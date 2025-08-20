#include <iostream>
#include <string>

// -------------------- Product --------------------
class Meal {
public:
    std::string drink;
    std::string mainDish;

    void show() const {
        std::cout << "Meal: " << mainDish << " + " << drink << "\n";
    }
};

// -------------------- Builder Interface --------------------
class MealBuilder {
public:
    virtual void buildDrink() = 0;
    virtual void buildMainDish() = 0;
    virtual Meal getMeal() = 0;
    virtual ~MealBuilder() = default;
};

// -------------------- Concrete Builders --------------------
class VegMealBuilder : public MealBuilder {
    Meal meal;
public:
    void buildDrink() override { meal.drink = "Juice"; }
    void buildMainDish() override { meal.mainDish = "Salad"; }
    Meal getMeal() override { return meal; }
};

class NonVegMealBuilder : public MealBuilder {
    Meal meal;
public:
    void buildDrink() override { meal.drink = "Soda"; }
    void buildMainDish() override { meal.mainDish = "Chicken"; }
    Meal getMeal() override { return meal; }
};

// -------------------- Director --------------------
class MealDirector {
public:
    Meal createMeal(MealBuilder& builder) {
        builder.buildDrink();
        builder.buildMainDish();
        return builder.getMeal();
    }
};

// -------------------- Client --------------------
int main() {
    MealDirector director;

    VegMealBuilder vegBuilder;
    Meal vegMeal = director.createMeal(vegBuilder);
    vegMeal.show();

    NonVegMealBuilder nonVegBuilder;
    Meal nonVegMeal = director.createMeal(nonVegBuilder);
    nonVegMeal.show();

    return 0;
}
