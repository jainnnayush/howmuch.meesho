import pandas as pd
import numpy as np

df = pd.read_csv('/mnt/data/meesho_sku_sales_data_60_days.csv')


base_prices = df.groupby('SKU')['Base_Price'].first().values
base_demand = df.groupby('SKU')['Predicted_Demand'].mean().values


unit_cost = df.groupby('SKU')['Unit_Cost'].mean().values if 'Unit_Cost' in df.columns else base_prices * 0.6


price_elasticity = df.groupby('SKU')['Price_Elasticity'].mean().values if 'Price_Elasticity' in df.columns else 1.5


discount_range = np.arange(0, 0.5, 0.005)


event_multiplier = 1.5


predicted_inventory = df.groupby('SKU')['Inventory_Level'].first().values


num_simulations = 20000  
optimal_discounts = []


def calculate_demand(base_demand, base_price, discounted_price, price_elasticity, event_multiplier):
    
    demand = base_demand * (base_price / discounted_price) ** price_elasticity * event_multiplier
    
    noise = np.random.normal(0, 0.05, size=len(demand))  
    demand = demand * (1 + noise)
    return np.maximum(demand, 0)  


for _ in range(num_simulations):
    
    discounts = np.random.choice(discount_range, size=len(base_prices))
    discounted_prices = base_prices * (1 - discounts)
    
    
    demand = calculate_demand(base_demand, base_prices, discounted_prices, price_elasticity, event_multiplier)
    
    
    demand = np.minimum(demand, predicted_inventory)
    
    
    revenue = np.sum(demand * discounted_prices)
    profit = np.sum(demand * (discounted_prices - unit_cost))  
    
    
    optimal_discounts.append((profit, revenue, discounts))

optimal_profit, optimal_revenue, best_discounts = max(optimal_discounts, key=lambda x: x[0])


print("Optimal Profit based on forecasted inventory:", optimal_profit)
print("Optimal Revenue based on forecasted inventory:", optimal_revenue)
print("Best Discounts for each SKU:", best_discounts)


print("\nRecommended Strategy:")
for i, discount in enumerate(best_discounts):
    print(f"SKU {i+1}: Apply a discount of {discount*100:.2f}%")
