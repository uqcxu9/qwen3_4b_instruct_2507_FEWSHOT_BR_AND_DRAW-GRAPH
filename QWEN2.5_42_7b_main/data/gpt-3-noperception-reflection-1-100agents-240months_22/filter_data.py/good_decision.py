import pickle as pkl
import numpy as np
import pandas as pd
import os

data_path = r'/workspace/QWEN2.5_42_7b_main/data/gpt-3-noperception-reflection-1-100agents-240months_22'

with open(os.path.join(data_path, 'dense_log.pkl'), 'rb', buffering=1024*1024*10) as f:
    dense_log = pkl.load(f)
states = dense_log['states']
actions = dense_log['actions']
periodic_tax = dense_log['PeriodicTax']

print(f"Total timesteps: {len(states)}")
print(f"Actions length: {len(actions)}")
print(f"Periodic_tax length: {len(periodic_tax)}")
class DummyUnpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if 'ai_economist' in module:
            return type(name, (), {})
        return super().find_class(module, name)

env_file = f'{data_path}/env_240.pkl'
with open(env_file, "rb") as f:
    env = DummyUnpickler(f).load()

prices = list(env.world.price)
print(f"Price data length: {len(prices)}")


A = 1
num_labor_hours = 168

def calculate_dpi(t, agent_id_str, states, periodic_tax):
    income = states[t][agent_id_str]['income']['Coin']
    lump_sum = periodic_tax[t].get(agent_id_str, {}).get('lump_sum', 0)
    tax_paid = periodic_tax[t].get(agent_id_str, {}).get('tax_paid', 0)
    return income + lump_sum - tax_paid

def calculate_monthly_gdp(t, states, actions, prices):
    monthly_supply = 0
    for agent_id, action in actions[t].items():
        if agent_id == 'p':
            continue
        if isinstance(action, dict):
            labor = int(action.get('SimpleLabor', 0))
        elif isinstance(action, (list, tuple)) and len(action) >= 1:
            labor = int(action[0])
        else:
            labor = 0
        monthly_supply += labor * num_labor_hours * A
    return monthly_supply * prices[t]

def calculate_yearly_wage_inflation(year, states, max_t):
    year_start = (year - 1) * 12
    year_end = year * 12
    prev_start = (year - 2) * 12
    prev_end = (year - 1) * 12
    
    if year_end > max_t or prev_start < 0:
        return None
    
    def avg_wage(start, end):
        wages = []
        for t in range(start, end):
            for aid, st in states[t].items():
                if aid == "p" or not isinstance(st, dict):
                    continue
                wage = st.get("skill", None)
                if wage is not None:
                    wages.append(float(wage))
        return np.mean(wages) if wages else None
    
    w_curr = avg_wage(year_start, year_end)
    w_prev = avg_wage(prev_start, prev_end)
    
    if w_curr is None or w_prev is None or w_prev <= 1e-9:
        return None
    
    return (w_curr / w_prev - 1) * 100

# Calculate yearly unemployment rates
def calculate_yearly_unemployment_rate(year, states, max_t):
    year_start = (year - 1) * 12
    year_end = year * 12

    if year_end > max_t:
        return None

    monthly_rates = []
    for t in range(year_start, year_end):
        unemployed = 0
        total = 0
        for aid, state in states[t].items():
            if aid == "p" or not isinstance(state, dict):
                continue
            total += 1
            job = state.get("endogenous", {}).get("job")
            if job == "Unemployment":
                unemployed += 1
        if total > 0:
            monthly_rates.append(unemployed / total)

    return np.mean(monthly_rates) if monthly_rates else None


# Calculate macro indicators + save yearly statistics
print("\n Step 1: Filter months with good macro performance...")

max_t = min(len(states), len(periodic_tax), len(actions), len(prices))
macro_good_months = []
year_stats = {}


# Define constraint thresholds
GDP_GROWTH_MIN = -3.0
GDP_GROWTH_MAX = 8.0
WAGE_INFLATION_MIN = -1.5
WAGE_INFLATION_MAX = 4.0
UNEMPLOYMENT_MIN = 0.03
UNEMPLOYMENT_MAX = 0.15

# Business cycle classification thresholds (data-driven quantiles)
gdp_growth_all_years = []

for year in range(2, 21):
    year_start_month = (year - 1) * 12
    year_end_month = year * 12

    if year_end_month > max_t:
        break

    curr_year_gdp = sum(calculate_monthly_gdp(t, states, actions, prices)
                        for t in range(year_start_month, year_end_month))
    prev_year_gdp = sum(calculate_monthly_gdp(t, states, actions, prices)
                        for t in range(year_start_month-12, year_start_month))

    if prev_year_gdp > 0:
        gdp_growth = (curr_year_gdp - prev_year_gdp) / prev_year_gdp * 100
        gdp_growth_all_years.append(gdp_growth)

RECESSION_THRESHOLD = np.percentile(gdp_growth_all_years, 25)
BOOM_THRESHOLD = np.percentile(gdp_growth_all_years, 75)
for year in range(2, 21):
    year_start_month = (year - 1) * 12
    year_end_month = year * 12
    
    if year_end_month > max_t:
        break
    
    # === Yearly GDP growth ===
    curr_year_gdp = sum(calculate_monthly_gdp(t, states, actions, prices) 
                       for t in range(year_start_month, year_end_month))
    prev_year_gdp = sum(calculate_monthly_gdp(t, states, actions, prices) 
                       for t in range(year_start_month-12, year_start_month))
    
    if prev_year_gdp > 0:
        gdp_growth = (curr_year_gdp - prev_year_gdp) / prev_year_gdp * 100
    else:
        continue
    
    # === Yearly price inflation ===
    curr_avg_price = np.mean([prices[t] for t in range(year_start_month, year_end_month)])
    prev_avg_price = np.mean([prices[t] for t in range(year_start_month-12, year_start_month)])
    price_inflation = (curr_avg_price - prev_avg_price) / prev_avg_price * 100
    
    wage_inflation = calculate_yearly_wage_inflation(year, states, max_t)
    yearly_unemployment = calculate_yearly_unemployment_rate(year, states, max_t)
    # Business cycle classification (based on potential GDP deviation)
    # Boom: GDP growth > μ + 2σ = 2.67%
    # Normal: GDP growth in [-1%, 2.67%]
    # Recession: Real economic contraction (GDP growth < -1%)
    if gdp_growth < RECESSION_THRESHOLD:
        macro_state = "recession"
    elif gdp_growth > BOOM_THRESHOLD:
        macro_state = "boom"
    else:
        macro_state = "normal"
    
    # Save yearly statistics
    year_stats[year] = {
        'gdp_growth': gdp_growth,
        'price_inflation': price_inflation,
        'wage_inflation': wage_inflation,
        'yearly_unemployment': yearly_unemployment,
        'macro_state': macro_state
    }
    
# === Yearly macro constraints (including unemployment) ===
    gdp_good = (GDP_GROWTH_MIN <= gdp_growth <= GDP_GROWTH_MAX)
    wage_inflation_good = (wage_inflation is not None and
                        WAGE_INFLATION_MIN <= wage_inflation <= WAGE_INFLATION_MAX)
    unemployment_good = (yearly_unemployment is not None and
                        UNEMPLOYMENT_MIN <= yearly_unemployment <= UNEMPLOYMENT_MAX)

    year_macro_good = gdp_good and wage_inflation_good and unemployment_good
    
    # Print yearly info
    status = "PASS" if year_macro_good else "FAIL"
    wage_str = f"{wage_inflation:.2f}%" if wage_inflation is not None else "N/A"
    unemp_str = f"{yearly_unemployment*100:.2f}%" if yearly_unemployment is not None else "N/A"
    print(f"Year {year} {status}: GDP={gdp_growth:.2f}%, "
        f"WageInfl={wage_str}, Unemp={unemp_str}, State={macro_state}")
    
# If yearly macro conditions pass, keep all months in that year
    if year_macro_good:
        for t in range(year_start_month, year_end_month):
            macro_good_months.append(t)

print(f"\n Found {len(macro_good_months)} months with good macro performance")
print(f"   (Yearly constraints: GDP in [{GDP_GROWTH_MIN}%,{GDP_GROWTH_MAX}%], "
      f"WageInfl in [{WAGE_INFLATION_MIN}%,{WAGE_INFLATION_MAX}%])")
print(f"   (Yearly constraint: Unemployment in [{UNEMPLOYMENT_MIN*100}%,{UNEMPLOYMENT_MAX*100}%])")


# Extract good micro decisions from macro-good months
print("\n Step 2: Extract good micro decisions from macro-good months...")

good_decisions = []

# Filter statistics
filter_stats = {
    'total': 0,
    'pass_budget': 0,
    'pass_all': 0
}

# Iterate through macro-good months
for t in macro_good_months:
    if t == 0:
        continue
    
    current_year = (t // 12) + 1
    
    # Get macro indicators from year_stats
    stats = year_stats.get(current_year, None)
    if stats is not None:
        gdp_growth = stats['gdp_growth']
        price_inflation = stats['price_inflation']
        wage_inflation = stats['wage_inflation']
        yearly_unemployment = stats['yearly_unemployment']
        macro_state = stats['macro_state']
    else:
        continue
    
    for agent_id in range(100):
        agent_id_str = str(agent_id)
        filter_stats['total'] += 1
        
        if agent_id_str not in states[t]:
            continue
        
        # Extract monthly data
        curr_consumption = states[t][agent_id_str]['consumption']['Coin']
        prev_consumption = states[t-1][agent_id_str]['consumption']['Coin']
        curr_income = states[t][agent_id_str]['income']['Coin']
        prev_income = states[t-1][agent_id_str]['income']['Coin']
        curr_wealth = states[t][agent_id_str]['inventory']['Coin']
        
        curr_tax = periodic_tax[t].get(agent_id_str, {}).get('tax_paid', 0)
        prev_tax = periodic_tax[t-1].get(agent_id_str, {}).get('tax_paid', 0)
        curr_lump = periodic_tax[t].get(agent_id_str, {}).get('lump_sum', 0)
        prev_lump = periodic_tax[t-1].get(agent_id_str, {}).get('lump_sum', 0)
        
        curr_dpi = curr_income + curr_lump - curr_tax
        prev_dpi = prev_income + prev_lump - prev_tax
        
        # Constraint 1: Budget constraint (physical constraint)
        if curr_consumption > curr_wealth + curr_income + 100:
            continue
        filter_stats['pass_budget'] += 1
        
        filter_stats['pass_all'] += 1
        
        # Extract decision
        job = states[t][agent_id_str].get('endogenous', {}).get('job')
        work_decision = 0.0 if job == "Unemployment" else 1.0
        
        if agent_id_str in actions[t]:
            action_data = actions[t][agent_id_str]
            if isinstance(action_data, dict):
                consumption_idx = action_data.get('SimpleConsumption', 25)
            elif isinstance(action_data, (list, tuple)) and len(action_data) >= 2:
                consumption_idx = action_data[1]
            else:
                consumption_idx = 25
            consumption_prop = consumption_idx * 0.02
        else:
            consumption_prop = 0.5
        
        # Save decision
        good_decisions.append({
            'timestep': t,
            'year': current_year,
            'agent_id': agent_id,
            'prev_consumption': prev_consumption,
            'curr_consumption': curr_consumption,
            'prev_income': prev_income,
            'curr_income': curr_income,
            'curr_wealth': curr_wealth,
            'prev_dpi': prev_dpi,
            'curr_dpi': curr_dpi,
            'work_decision': work_decision,
            'consumption_prop': consumption_prop,
            'macro_state': macro_state,
            'gdp_growth': gdp_growth,
            'price_inflation': price_inflation,
            'wage_inflation': wage_inflation,
            'yearly_unemployment': yearly_unemployment,
        })

print(f"\n Extracted {len(good_decisions)} good decisions")
print(f"\n Filter statistics:")
print(f"   Total candidates: {filter_stats['total']}")
print(f"   Pass budget constraint: {filter_stats['pass_budget']} ({filter_stats['pass_budget']/max(filter_stats['total'],1)*100:.1f}%)")
print(f"   Final pass: {filter_stats['pass_all']} ({filter_stats['pass_all']/max(filter_stats['total'],1)*100:.1f}%)")


# ========== 5. Balanced sampling + Save ==========
if len(good_decisions) == 0:
    print("\n Warning: No decisions found that meet the criteria!")
    print("Suggestions:")
    print(f"  1. Relax wage inflation constraint (current: [{WAGE_INFLATION_MIN}%, {WAGE_INFLATION_MAX}%])")
else:
    df_good = pd.DataFrame(good_decisions)
    
    # Balanced sampling
    print("\n Performing employment/unemployment balanced sampling...")
    
    employed_decisions = df_good[df_good['work_decision'] == 1.0].copy()
    unemployed_decisions = df_good[df_good['work_decision'] == 0.0].copy()
    
    print(f"Original data: Employed {len(employed_decisions)}, Unemployed {len(unemployed_decisions)}")
    print(f"Original unemployment rate: {len(unemployed_decisions)/len(df_good)*100:.1f}%")
    
    target_unemployed_ratio = 0.15
    
    if len(unemployed_decisions) > 0 and len(employed_decisions) > 0:
        original_ratio = len(unemployed_decisions) / len(df_good)
        
        if original_ratio >= target_unemployed_ratio:
            print(f" Unemployment rate meets target ({original_ratio*100:.1f}% >= {target_unemployed_ratio*100:.1f}%)")
        else:
            target_employed_count = int(len(unemployed_decisions) / target_unemployed_ratio * (1 - target_unemployed_ratio))
            
            if len(employed_decisions) > target_employed_count:
               
                employed_sampled = employed_decisions.sample(n=target_employed_count, random_state=42)
            else:
                employed_sampled = employed_decisions
            
            df_good = pd.concat([employed_sampled, unemployed_decisions], ignore_index=True)
            
            print(f" After balancing: Employed {len(employed_sampled)}, Unemployed {len(unemployed_decisions)}")
            print(f"   Unemployment ratio: {len(unemployed_decisions)/len(df_good)*100:.1f}%")
    else:
        print(" Cannot balance: Missing employed or unemployed decisions")
    
    # Statistics
    print("\n" + "="*60)
    print(" Data Quality Check")
    print("="*60)
    print(f"Source years: {sorted(df_good['year'].unique())}")
    print(f"Total decisions: {len(df_good)}")
    print(f"Employed ratio: {(df_good['work_decision']==1).sum()/len(df_good)*100:.1f}%")
    print(f"Unemployed ratio: {(df_good['work_decision']==0).sum()/len(df_good)*100:.1f}%")
    
    # Macro state distribution
    print(f"\nMacro state distribution:")
    print(df_good['macro_state'].value_counts())
    
    # Macro indicators statistics
    print(f"\nMacro indicators statistics:")
    print(f"  GDP growth: mean={df_good['gdp_growth'].mean():.2f}%, "
          f"range=[{df_good['gdp_growth'].min():.2f}%, {df_good['gdp_growth'].max():.2f}%]")
    wage_valid = df_good['wage_inflation'].dropna()
    if len(wage_valid) > 0:
        print(f"  Wage inflation: mean={wage_valid.mean():.2f}%, "
              f"range=[{wage_valid.min():.2f}%, {wage_valid.max():.2f}%]")
    
    # Save
    output_path = '/workspace/QWEN2.5_42_7b_main/data/gpt-3-noperception-reflection-1-100agents-240months_merge/good_decisions.csv_2'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_good.to_csv(output_path, index=False)
    
    print(f"\n Results saved to: {output_path}")
    
    # Print filter criteria summary
    print("\n" + "="*60)
    print(" Filter Criteria Summary")
    print("="*60)
    print("[Macro Level - Yearly]")
    print(f"  - GDP growth in [{GDP_GROWTH_MIN}%, {GDP_GROWTH_MAX}%]")
    print(f"  - Wage inflation in [{WAGE_INFLATION_MIN}%, {WAGE_INFLATION_MAX}%]")
    print(f"  - Yearly unemployment in [{UNEMPLOYMENT_MIN*100}%, {UNEMPLOYMENT_MAX*100}%]")
    print("[Micro Level - Agent]")
    print(f"  - Budget constraint: Consumption <= Wealth + Income + 100")
    print("[Business Cycle Classification (based on GDP growth quantiles)]")
    print(f"  - Recession: GDP growth < {RECESSION_THRESHOLD:.2f}% (bottom 25%)")
    print(f"  - Normal: GDP growth in [{RECESSION_THRESHOLD:.2f}%, {BOOM_THRESHOLD:.2f}%] (middle 50%)")
    print(f"  - Boom: GDP growth > {BOOM_THRESHOLD:.2f}% (top 25%)")
    print(f"  Yearly unemployment: mean={df_good['yearly_unemployment'].mean()*100:.2f}%, "
      f"range=[{df_good['yearly_unemployment'].min()*100:.2f}%, {df_good['yearly_unemployment'].max()*100:.2f}%]")