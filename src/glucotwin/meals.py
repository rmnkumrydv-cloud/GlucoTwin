"""Indian meal library for the what-if simulator.

IMPORTANT: values are APPROXIMATE per-serving estimates for typical home preparation, meant
for demonstration. Before the final submission, verify or replace them with values computed
from the Indian Food Composition Tables (IFCT 2017, ICMR-National Institute of Nutrition,
Hyderabad) or NIN recipe data, and cite the source in the README.

The model itself learns glucose responses from macronutrients (carbs, protein, fat, fiber),
not from dish names, so any dish can be simulated once its macros are known.
"""
from __future__ import annotations

# name: (serving description, kcal, carbs g, protein g, fat g, fiber g)
INDIAN_MEALS: dict[str, tuple[str, float, float, float, float, float]] = {
    "Idli": ("2 pieces", 130, 26, 4, 0.5, 1.5),
    "Plain dosa": ("1 medium", 170, 28, 4, 5, 1),
    "Masala dosa": ("1 medium", 300, 45, 6, 11, 3),
    "Pesarattu": ("1 medium", 180, 25, 9, 5, 4),
    "Upma": ("1 cup", 250, 38, 6, 8, 3),
    "Poha": ("1 cup", 250, 42, 5, 7, 2),
    "Ven pongal": ("1 cup", 280, 38, 8, 10, 2),
    "Medu vada": ("2 pieces", 280, 28, 8, 15, 3),
    "Aloo paratha": ("1 piece", 290, 40, 6, 11, 4),
    "Sambar": ("1 katori", 110, 15, 5, 3, 4),
    "Coconut chutney": ("2 tbsp", 70, 3, 1, 6, 2),
    "White rice (cooked)": ("1 cup", 240, 53, 4.5, 0.5, 0.7),
    "Brown rice (cooked)": ("1 cup", 215, 45, 5, 1.8, 3.5),
    "Foxtail millet (cooked)": ("1 cup", 200, 40, 5, 1.5, 3.5),
    "Ragi mudde": ("1 ball", 165, 36, 3.5, 0.7, 5.5),
    "Phulka / roti": ("1 piece", 85, 17, 3, 0.5, 2.5),
    "Jowar roti": ("1 piece", 100, 21, 3, 1, 2.5),
    "Toor dal": ("1 katori", 150, 20, 8, 4, 5),
    "Rajma curry": ("1 katori", 180, 24, 9, 5, 7),
    "Chana masala": ("1 katori", 200, 27, 9, 6, 7),
    "Mixed vegetable sabzi": ("1 katori", 120, 12, 3, 7, 4),
    "Chicken curry": ("1 katori", 250, 6, 22, 15, 1),
    "Boiled egg": ("1 egg", 70, 0.5, 6, 5, 0),
    "Curd": ("1 katori", 60, 4.5, 3.5, 3, 0),
    "Curd rice": ("1 cup", 250, 38, 6, 7, 1),
    "Chicken biryani": ("1 plate", 500, 60, 22, 18, 2),
    "Veg biryani": ("1 plate", 420, 65, 9, 13, 4),
    "Sprouts salad": ("1 cup", 100, 15, 7, 1, 5),
    "Banana": ("1 medium", 105, 27, 1.3, 0.4, 3),
    "Samosa": ("1 piece", 260, 30, 4, 14, 2.5),
    "Gulab jamun": ("2 pieces", 300, 45, 4, 12, 0.5),
    "Masala chai with sugar": ("1 cup", 90, 13, 2.5, 3, 0),
    "Filter coffee with sugar": ("1 cup", 80, 11, 2.5, 3, 0),
}


def compose_meal(items: dict[str, float]) -> dict[str, float]:
    """items: {dish name: number of servings} -> total macros."""
    total = {"meal_kcal": 0.0, "carbs": 0.0, "protein": 0.0, "fat": 0.0, "fiber": 0.0}
    for name, servings in items.items():
        if servings <= 0:
            continue
        if name not in INDIAN_MEALS:
            raise KeyError(f"Unknown dish '{name}'. Options: {sorted(INDIAN_MEALS)}")
        _, kcal, c, p, f, fb = INDIAN_MEALS[name]
        total["meal_kcal"] += kcal * servings
        total["carbs"] += c * servings
        total["protein"] += p * servings
        total["fat"] += f * servings
        total["fiber"] += fb * servings
    return {k: round(v, 1) for k, v in total.items()}


def meal_table() -> list[dict]:
    return [
        {"dish": k, "serving": v[0], "kcal": v[1], "carbs_g": v[2], "protein_g": v[3], "fat_g": v[4], "fiber_g": v[5]}
        for k, v in INDIAN_MEALS.items()
    ]
