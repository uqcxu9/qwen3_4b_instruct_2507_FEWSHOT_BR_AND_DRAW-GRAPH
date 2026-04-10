from typing import Optional
import argparse
import fire
import os
import sys
import pandas as pd
import ai_economist.foundation as foundation
import numpy as np
import matplotlib.pyplot as plt
import yaml
from time import time
from collections import defaultdict
import re
from simulate_utils import *
import pickle as pkl
from itertools import product
from dateutil.relativedelta import relativedelta
GOOD_DECISIONS_DF = None
GOOD_DECISIONS_STATS = None

def load_good_decisions():
    """延迟加载好决策数据"""
    global GOOD_DECISIONS_DF, GOOD_DECISIONS_STATS
    if GOOD_DECISIONS_DF is None:
        csv_path = '/workspace/QWEN2.5_42_7b_main/data/gpt-3-noperception-reflection-1-100agents-240months_merge/good_decisions_merged_4b.csv'
        if os.path.exists(csv_path):
            GOOD_DECISIONS_DF = pd.read_csv(csv_path)

            GOOD_DECISIONS_DF['curr_dpi_safe'] = GOOD_DECISIONS_DF['curr_dpi'].clip(lower=1e-6)
            GOOD_DECISIONS_DF['buffer_ratio'] = (
                (GOOD_DECISIONS_DF['curr_wealth'] + GOOD_DECISIONS_DF['curr_dpi_safe'])
                / GOOD_DECISIONS_DF['curr_dpi_safe']
            )

            # ====== Stats for offline alignment ======
            income_safe = GOOD_DECISIONS_DF['curr_income'].clip(lower=1e-6)
            wealth_safe = GOOD_DECISIONS_DF['curr_wealth'].clip(lower=1e-6)
            log_income = np.log(income_safe)
            log_wealth = np.log(wealth_safe)

            GOOD_DECISIONS_STATS = {
                'log_income_mean': log_income.mean(),
                'log_income_std': log_income.std() + 1e-8,
                'log_wealth_mean': log_wealth.mean(),
                'log_wealth_std': log_wealth.std() + 1e-8,
                # 与离线 merged 数据保持一致：按 real_gdp_growth 分位数划分 macro_state
                'recession_threshold': GOOD_DECISIONS_DF['real_gdp_growth'].quantile(0.25),
                'boom_threshold': GOOD_DECISIONS_DF['real_gdp_growth'].quantile(0.75),
            }

            print(f"✅ 加载了 {len(GOOD_DECISIONS_DF)} 个好决策示例")
        else:
            print(f"⚠️ 未找到好决策文件: {csv_path}")
            GOOD_DECISIONS_DF = pd.DataFrame()
            GOOD_DECISIONS_STATS = None
    return GOOD_DECISIONS_DF, GOOD_DECISIONS_STATS

with open('config.yaml', "r") as f:
    run_configuration = yaml.safe_load(f)
env_config = run_configuration.get('env')

def get_economic_state(env):
    """判断当前经济状态（基于 rolling GDP proxy 的增长）"""
    actions = env.dense_log.get('actions', [])
    states = env.dense_log.get('states', [])
    prices = getattr(env.world, "price", [])

    max_l = env._components_dict['SimpleLabor'].num_labor_hours

    def _get_labor_action(action_map, agent_id: str) -> int:
        last_action = action_map.get(agent_id, {})
        if isinstance(last_action, dict):
            return int(last_action.get('SimpleLabor', 0))
        if isinstance(last_action, (list, tuple)) and len(last_action) >= 1:
            return int(last_action[0])
        return 0

    def _monthly_gdp_proxy(t: int) -> float:
        if t < 0 or t >= len(actions) or t >= len(prices):
            return 0.0
        supply = 0.0
        for i in range(env.num_agents):
            aid = str(i)
            labor = _get_labor_action(actions[t], aid)
            if labor <= 0:
                continue
            supply += labor * max_l * 1.0
        return supply * float(prices[0])

    def _monthly_avg_skill(t: int) -> float:
        if t < 0 or t >= len(states) or not isinstance(states[t], dict):
            return 0.0
        skills = []
        for i in range(env.num_agents):
            st = states[t].get(str(i), {})
            if isinstance(st, dict):
                s = st.get('skill', None)
                if s is not None:
                    skills.append(float(s))
        return float(np.mean(skills)) if skills else 0.0

    def _rolling_annual_wage_growth() -> float:
        # rolling annual wage growth = (avg skill last 12m - avg skill prev 12m) / avg skill prev 12m
        if len(states) < 24:
            return 0.0
        last12 = np.mean([_monthly_avg_skill(t) for t in range(len(states) - 12, len(states))])
        prev12 = np.mean([_monthly_avg_skill(t) for t in range(len(states) - 24, len(states) - 12)])
        return float((last12 - prev12) / (prev12 + 1e-8))

    # 至少需要 24 个月：最近 12 个月 vs 前 12 个月
    if env.world.timestep < 24 or len(actions) < 24:
        if len(actions) == 0:
            return "Normal", 0.0, 0.0, 0.0

        monthly_unemp_rates = []
        for t in range(len(actions)):
            employed_t = sum(1 for i in range(env.num_agents) if _get_labor_action(actions[t], str(i)) > 0)
            unemp_t = 1 - employed_t / max(env.num_agents, 1)
            monthly_unemp_rates.append(unemp_t)

        unemployment_rate = float(np.mean(monthly_unemp_rates)) if monthly_unemp_rates else 0.0
        return "Normal", unemployment_rate, 0.0, 0.0

    last12 = sum(_monthly_gdp_proxy(t) for t in range(len(actions) - 12, len(actions)))
    prev12 = sum(_monthly_gdp_proxy(t) for t in range(len(actions) - 24, len(actions) - 12))
    real_gdp_growth = (last12 - prev12) / (prev12 + 1e-8) * 100

    monthly_unemp_rates = []
    for t in range(max(0, len(actions) - 12), len(actions)):
        employed_t = sum(1 for i in range(env.num_agents) if _get_labor_action(actions[t], str(i)) > 0)
        unemp_t = 1 - employed_t / max(env.num_agents, 1)
        monthly_unemp_rates.append(unemp_t)

    unemployment_rate = float(np.mean(monthly_unemp_rates)) if monthly_unemp_rates else 0.0
    wage_growth = _rolling_annual_wage_growth()

    _, good_stats = load_good_decisions()
    recession_threshold = good_stats['recession_threshold'] if good_stats is not None else -2.5067
    boom_threshold = good_stats['boom_threshold'] if good_stats is not None else 0.6034

    if real_gdp_growth <= recession_threshold:
        return "Recession", unemployment_rate, real_gdp_growth, wage_growth
    if real_gdp_growth >= boom_threshold:
        return "Boom", unemployment_rate, real_gdp_growth, wage_growth
    return "Normal", unemployment_rate, real_gdp_growth, wage_growth

def gpt_actions(env, obs, dialog_queue, dialog4ref_queue, gpt_path, gpt_error, total_cost, model_type='qwen'):
    if not os.path.exists(gpt_path):
        os.makedirs(gpt_path)
    curr_rates = obs['p']['PeriodicBracketTax-curr_rates']
    current_time = world_start_time + relativedelta(months=env.world.timestep)
    current_time = current_time.strftime('%Y.%m')
    for idx in range(env.num_agents):
        this_agent = env.get_agent(str(idx))
        skill = this_agent.state['skill']
        wealth = this_agent.inventory['Coin']
        consumption = this_agent.consumption['Coin']
        interest_rate = env.world.interest_rate[-1]
        price = env.world.price[-1]
        tax_paid = obs['p'][f'p{idx}']['PeriodicBracketTax-tax_paid']
        lump_sum = obs['p'][f'p{idx}']['PeriodicBracketTax-lump_sum']
        max_l = env._components_dict['SimpleLabor'].num_labor_hours
        name = this_agent.endogenous['name']
        age = this_agent.endogenous['age']
        city = this_agent.endogenous['city']
        job = this_agent.endogenous['job']
        offer = this_agent.endogenous['offer']
        actions = env.dense_log['actions']
        states = env.dense_log['states']

        # ========== 🆕 Few-shot匹配（改进版：考虑上一期状态 + 宏观） ==========
        few_shot_examples = ""
        good_df, _ = load_good_decisions()
        if len(good_df) > 0 and env.world.timestep > 0:
            required_cols = [
                'prev_work_decision', 'macro_state', 'curr_income', 'curr_dpi',
                'curr_dpi_safe', 'curr_wealth', 'buffer_ratio',
                'work_decision', 'consumption_prop'
            ]
            missing_cols = [c for c in required_cols if c not in good_df.columns]
            if missing_cols:
                raise ValueError(f"good_decisions_merged.csv 缺少列: {missing_cols}")

            curr_income = this_agent.income['Coin']
            curr_dpi = curr_income + lump_sum - tax_paid
            curr_dpi_safe = max(curr_dpi, 1e-6)
            curr_buffer_ratio = (wealth + curr_dpi_safe) / curr_dpi_safe

            cold_start = (
                env.world.timestep <= 1 and
                curr_income <= 1e-6 and
                wealth <= 1e-6
            )

            economic_state, _, _, _ = get_economic_state(env)
            current_macro_state = economic_state.lower()

            # 上一期是否工作（agent 真实行为）
            last_action = actions[-1].get(str(idx), {}) if len(actions) > 0 else {}
            if isinstance(last_action, dict):
                prev_work_decision_agent = int(last_action.get('SimpleLabor', 0))
            elif isinstance(last_action, (list, tuple)) and len(last_action) >= 1:
                prev_work_decision_agent = int(last_action[0])
            else:
                prev_work_decision_agent = 0

            # 先按“上一期是否工作”分桶
            if prev_work_decision_agent == 0:
                state_candidates = good_df[
                    good_df['prev_work_decision'] == 0.0
                ].copy()
            else:
                state_candidates = good_df[
                    good_df['prev_work_decision'] == 1.0
                ].copy()

            # 如果状态桶太小，回退全库
            if len(state_candidates) < 10:
                state_candidates = good_df.copy()

            # 再按 macro_state
            if 'macro_state' in good_df.columns:
                macro_candidates = state_candidates[
                    state_candidates['macro_state'] == current_macro_state
                ].copy()
                if len(macro_candidates) >= 5:
                    state_candidates = macro_candidates

            if cold_start:
                # cold-start: 先限制在低 income / 低 wealth / 低 dpi 子池里
                q_income = state_candidates['curr_income'].quantile(0.30)
                q_wealth = state_candidates['curr_wealth'].quantile(0.30)
                q_dpi = state_candidates['curr_dpi_safe'].quantile(0.40)

                cold_candidates = state_candidates[
                    (state_candidates['curr_income'] <= max(q_income, 1.0)) &
                    (state_candidates['curr_wealth'] <= max(q_wealth, 1.0)) &
                    (state_candidates['curr_dpi_safe'] <= max(q_dpi, 1.0))
                ].copy()

                # 如果太少，逐步放宽，但仍然不直接回全库高状态样本
                if len(cold_candidates) < 10:
                    cold_candidates = state_candidates[
                        (state_candidates['curr_income'] <= max(q_income, 1.0)) &
                        (state_candidates['curr_dpi_safe'] <= max(q_dpi, 1.0))
                    ].copy()

                if len(cold_candidates) < 10:
                    cold_candidates = state_candidates[
                        state_candidates['curr_dpi_safe'] <= max(q_dpi, 1.0)
                    ].copy()

                if len(cold_candidates) >= 5:
                    state_candidates = cold_candidates

            else:
                # 常规阶段沿用原逻辑
                if curr_income > 1e-6:
                    income_candidates = state_candidates[
                        (state_candidates['curr_income'] >= curr_income * 0.5) &
                        (state_candidates['curr_income'] <= curr_income * 2.0)
                    ].copy()
                    if len(income_candidates) >= 10:
                        state_candidates = income_candidates

                dpi_candidates = state_candidates[
                    (state_candidates['curr_dpi_safe'] >= curr_dpi_safe * 0.5) &
                    (state_candidates['curr_dpi_safe'] <= curr_dpi_safe * 2.0)
                ].copy()
                if len(dpi_candidates) >= 10:
                    state_candidates = dpi_candidates

            candidates = state_candidates.copy()

            # if env.world.timestep == 1 and idx < 2:
            #     print(f"[DEBUG] t={env.world.timestep}, agent={idx}", flush=True)
            #     print(f"[DEBUG] len(good_df)={len(good_df)}", flush=True)
            #     print(f"[DEBUG] cold_start={cold_start}", flush=True)
            #     print(f"[DEBUG] prev_work_decision_agent={prev_work_decision_agent}", flush=True)
            #     print(f"[DEBUG] state_candidates_len={len(state_candidates)}", flush=True)
            #     print(f"[DEBUG] candidates_len={len(candidates)}", flush=True)

            if len(candidates) > 0:
                candidates = candidates.copy()
                curr_wealth_safe = max(wealth, 0.0)

                candidates['buffer_distance'] = (
                    np.log1p(candidates['buffer_ratio']) - np.log1p(curr_buffer_ratio)
                ).abs()

                candidates['dpi_distance'] = (
                    np.log1p(candidates['curr_dpi_safe']) - np.log1p(curr_dpi_safe)
                ).abs()

                candidates['wealth_distance'] = (
                    np.log1p(candidates['curr_wealth'].clip(lower=0.0)) - np.log1p(curr_wealth_safe)
                ).abs()

                if cold_start:
                    candidates['match_score'] = (
                        0.10 * candidates['buffer_distance'] +
                        0.45 * candidates['dpi_distance'] +
                        0.45 * candidates['wealth_distance']
                    )
                else:
                    candidates['match_score'] = (
                        0.4 * candidates['buffer_distance'] +
                        0.4 * candidates['dpi_distance'] +
                        0.2 * candidates['wealth_distance']
                    )

                candidates = candidates.sort_values('match_score')
                top_decisions = candidates.head(3).copy()

                # 多样性补丁：仅在 top-20 内找“反例”，避免引入非常远的例子
                if top_decisions['work_decision'].nunique() == 1:
                    search_pool = candidates.head(20)
                    alt_pool = search_pool[
                        (search_pool['work_decision'] != top_decisions['work_decision'].iloc[0]) &
                        (~search_pool.index.isin(top_decisions.index))
                    ]
                    if len(alt_pool) > 0:
                        top_decisions = pd.concat([top_decisions.iloc[:2], alt_pool.head(1)])

                top_decisions = top_decisions.sort_values('match_score')

                few_shot_examples = "\n\n**Examples of good economic decisions:**\n"
                for i, (_, row) in enumerate(top_decisions.iterrows(), 1):
                    few_shot_examples += (
                        f"Example {i}: The person did "
                        f"{'not work' if row['prev_work_decision'] == 0 else 'work'} in the previous month. "
                        f"At the time of making the current decision, they had income ${row['curr_income']:.0f}, "
                        f"DPI ${row['curr_dpi']:.0f}, wealth ${row['curr_wealth']:.0f}, "
                        f"and buffer ratio {row['buffer_ratio']:.2f}. "
                        f"They then chose to "
                        f"{'work' if row['work_decision'] == 1 else 'not work'} "
                        f"and set consumption to {row['consumption_prop']:.2f}.\n"
                    )
                # if env.world.timestep == 1 and idx < 2:
                #     print(f"[DEBUG] top_decisions_len={len(top_decisions)}", flush=True)
                #     print(f"[DEBUG] few_shot_len={len(few_shot_examples)}", flush=True)
                #     print(f"[DEBUG] few_shot_examples={few_shot_examples[:400]}", flush=True)
        # ========== 🆕 宏观信号注入 ==========
        macro_signal = ""
        if env.world.timestep > 0:
            economic_state, unemployment_rate, gdp_growth, wage_growth = get_economic_state(env)
            
            macro_signal = f"""
**Current Economic Indicators (for your reference):**
- Unemployment rate: {unemployment_rate*100:.1f}%
- Recent real GDP growth: {gdp_growth:.1f}% (rolling annual)
- Recent wage inflation: {wage_growth*100:.1f}% (rolling annual)

These short-term movements are normal parts of a standard economic cycle.

**Important:** Your own decision should depend mainly on your personal situation —  
including income stability, savings, prices, and basic needs — rather than 
following general trends. 
"""
        # ========== 宏观信号结束 ==========
        problem_prompt = f'''
                    You're {name}, a {age}-year-old individual living in {city}. As with all Americans, a portion of your monthly income is taxed by the federal government. This taxation system is tiered, income is taxed cumulatively within defined brackets, combined with a redistributive policy: after collection, the government evenly redistributes the tax revenue back to all citizens, irrespective of their earnings.
                    Now it's {current_time}.
                '''
        



        if job == 'Unemployment':
            job_prompt = f'''
                        In the previous month, you became unemployed and had no income. Now, you are invited to work as a(an) {offer} with monthly salary of ${skill*max_l:.2f}.
                    '''
        else:
            if skill >= states[-1][str(idx)]['skill']:
                job_prompt = f'''
                            In the previous month, you worked as a(an) {job}. If you continue working this month, your expected income will be ${skill*max_l:.2f}, which is increased compared to the last month due to the inflation of labor market.
                        '''
            else:
                job_prompt = f'''
                            In the previous month, you worked as a(an) {job}. If you continue working this month, your expected income will be ${skill*max_l:.2f}, which is decreased compared to the last month due to the deflation of labor market.
                        '''
        last_action = actions[-1].get(str(idx), {}) if len(actions) > 0 else {}
        if isinstance(last_action, dict):
            last_consumption_action = last_action.get('SimpleConsumption', 0)
        elif isinstance(last_action, (list, tuple)) and len(last_action) >= 2:
            last_consumption_action = last_action[1]
        else:
            last_consumption_action = 0

        if (consumption <= 0) and (len(actions) > 0) and (last_consumption_action > 0):
            consumption_prompt = f'''
                        Besides, you had no consumption due to shortage of goods.
                    '''
        else:
            consumption_prompt = f'''
                        Besides, your consumption was ${consumption:.2f}.
                    '''
        if env._components_dict['PeriodicBracketTax'].tax_model == 'us-federal-single-filer-2018-scaled':
            tax_prompt = f'''Your tax deduction amounted to ${tax_paid:.2f}. However, as part of the government's redistribution program, you received a credit of ${lump_sum:.2f}.
                            In this month, the government sets the brackets: {format_numbers(brackets)} and their corresponding rates: {format_numbers(curr_rates)}. Income earned within each bracket is taxed only at that bracket's rate.'''
        else:
            tax_prompt = f'''Your tax deduction amounted to ${tax_paid:.2f}. However, as part of the government's redistribution program, you received a credit of ${lump_sum:.2f}.
                            In this month, according to the optimal taxation theory, Saez Tax, the brackets are not changed: {format_numbers(brackets)} but the government has updated corresponding rates: {format_percentages(curr_rates)}. Income earned within each bracket is taxed only at that bracket's rate.'''
        if env.world.timestep == 0:
            price_prompt = f'''Meanwhile, in the consumption market, the average price of essential goods is now at ${price:.2f}.'''
        else:
            if price >= env.world.price[-2]:
                price_prompt = f'''Meanwhile, inflation has led to a price increase in the consumption market, with the average price of essential goods now at ${price:.2f}.'''
            else:
                price_prompt = f'''Meanwhile, deflation has led to a price decrease in the consumption market, with the average price of essential goods now at ${price:.2f}.'''
        job_prompt = prettify_document(job_prompt)
        obs_prompt = f'''
                        {problem_prompt} {job_prompt} {consumption_prompt} {tax_prompt} {price_prompt}
                        Your current savings account balance is ${wealth:.2f}. Interest rates, as set by your bank, stand at {interest_rate*100:.2f}%.{macro_signal}{few_shot_examples}
                        With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures on essential goods, keeping in mind good price?
                        Please share your decisions in a JSON format. The format should have two keys: 'work' (a value between 0 and 1 with intervals of 0.02, indicating the willingness or propensity to work) and 'consumption' (a value between 0 and 1 with intervals of 0.02, indicating the proportion of all your savings and income you intend to spend on essential goods).
                    '''
        obs_prompt = prettify_document(obs_prompt)
        dialog_queue[idx].append({'role': 'user', 'content': obs_prompt})
        dialog4ref_queue[idx].append({'role': 'user', 'content': obs_prompt})
        
    def action_check(actions):
        if len(actions) != 2:
            return False
        else:
            return (actions[0] >= 0) & (actions[0] <= 1) & (actions[1] >= 0) & (actions[1] <= 1)
    if env.world.timestep%3 == 0 and env.world.timestep > 0:
        results, cost = get_multiple_completion([list(dialogs)[:2] + list(dialog4ref)[-3:-1] + list(dialogs)[-1:] for dialogs, dialog4ref in zip(dialog_queue, dialog4ref_queue)], model_type=model_type)
        total_cost += cost
    else:
        results, cost = get_multiple_completion([list(dialogs) for dialogs in dialog_queue], model_type=model_type)
        total_cost += cost
    actions = {}
    for idx in range(env.num_agents):
        content = results[idx]
        try:
            extracted_actions = list(eval(content).values())
            if not action_check(extracted_actions):
                extracted_actions = [1, 0.5]
                gpt_error += 1
        except:
            extracted_actions = [1, 0.5]
            gpt_error += 1
        extracted_actions[0] = int(np.random.uniform() <= extracted_actions[0])
        extracted_actions[1] /= 0.02
        actions[str(idx)] = extracted_actions
        dialog_queue[idx].append({'role': 'assistant', 'content': f'{content}'})
        dialog4ref_queue[idx].append({'role': 'assistant', 'content': f'{content}'})
    actions['p'] = [0]
    for idx, agent_dialog in enumerate(dialog_queue):
        with open(f'''{gpt_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
            for dialog in list(agent_dialog)[-2:]:
                f.write(f'''>>>>>>>>>{dialog['role']}: {dialog['content']}\n''')
        
    if (env.world.timestep+1)%3 == 0:
        reflection_prompt = '''Given the previous quarter's economic environment, reflect on the labor, consumption, and financial markets, as well as their dynamics. What conclusions have you drawn?
        Your answer must be less than 200 words!'''
        reflection_prompt = prettify_document(reflection_prompt)
        for idx in range(env.num_agents):
            # dialog_queue[idx].append({'role': 'user', 'content': reflection_prompt})
            dialog4ref_queue[idx].append({'role': 'user', 'content': reflection_prompt})
        results, cost = get_multiple_completion([list(dialogs) for dialogs in dialog4ref_queue], temperature=0, max_tokens=200, model_type=model_type)
        total_cost += cost
        for idx in range(env.num_agents):
            content = results[idx]
            # dialog_queue[idx].append({'role': 'assistant', 'content': content})
            dialog4ref_queue[idx].append({'role': 'assistant', 'content': content})
        
        for idx, agent_dialog in enumerate(dialog4ref_queue):
             with open(f'''{gpt_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
                for dialog in list(agent_dialog)[-2:]:
                    f.write(f'''>>>>>>>>>{dialog['role']}: {dialog['content']}\n''')
    return actions, gpt_error, total_cost

def complex_actions(env, obs, beta=0.1, gamma=0.1, h=1):

    def consumption_len(price, wealth, curr_income, last_income, interest_rate):
        c = (price/(1e-8+wealth+curr_income))**beta
        c = min(max(c//0.02, 0), 50)
        return c
    def consumption_cats(price, wealth, curr_income, last_income, interest_rate):
        h1 = h / (1 + interest_rate)
        g = curr_income/(last_income+1e-8) - 1
        d = wealth/(last_income+1e-8) - h1
        c = 1 + (d - h1*g)/(1 + g + 1e-8)
        c = min(max(c*curr_income/(wealth+curr_income+1e-8)//0.02, 0), 50)
        return c
    def work_income_wealth(price, wealth, curr_income, last_income, expected_income, interest_rate):
        return int(np.random.uniform() < (curr_income/(wealth*(1 + interest_rate)+1e-8))**gamma)
    
    consumption_funs = [consumption_len, consumption_cats]
    work_funs = [work_income_wealth]

    actions = {}
    for idx in range(env.num_agents):
        this_agent = env.get_agent(str(idx))
        price = env.world.price[-1]
        wealth = this_agent.inventory['Coin']
        max_l = env._components_dict['SimpleLabor'].num_labor_hours
        max_income = max_l * this_agent.state['skill']
        last_income = this_agent.income['Coin']
        expected_income = max_l * this_agent.state['expected skill']
        interest_rate = env.world.interest_rate[-1]
        if 'consumption_fun_idx' not in this_agent.endogenous:
            this_agent.endogenous['consumption_fun_idx'] = np.random.choice(range(len(consumption_funs)))
        if 'work_fun_idx' not in this_agent.endogenous:
            this_agent.endogenous['work_fun_idx'] = np.random.choice(range(len(work_funs)))
        work_fun = work_funs[this_agent.endogenous['work_fun_idx']]
        l = work_fun(price, wealth, max_income, last_income, expected_income, interest_rate)
        curr_income = l * max_income
        consumption_fun = consumption_funs[this_agent.endogenous['consumption_fun_idx']]
        c = consumption_fun(price, wealth, curr_income, last_income, interest_rate)
        actions[str(idx)] = [l, c]
    actions['p'] = [0]
    return actions
    

def main(policy_model='gpt', num_agents=100, episode_length=240, dialog_len=3, beta=0.1, gamma=0.1, h=1, max_price_inflation=0.1, max_wage_inflation=0.05, model_type='qwen'):
    env_config['n_agents'] = num_agents
    env_config['episode_length'] = episode_length
    if policy_model == 'gpt':
        total_cost = 0
        env_config['flatten_masks'] = False
        env_config['flatten_observations'] = False
        env_config['components'][0]['SimpleLabor']['scale_obs'] = False
        env_config['components'][1]['PeriodicBracketTax']['scale_obs'] = False
        env_config['components'][3]['SimpleSaving']['scale_obs'] = False
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation
        
        gpt_error = 0
        from collections import deque
        dialog_queue = [deque(maxlen=dialog_len) for _ in range(env_config['n_agents'])]
        dialog4ref_queue = [deque(maxlen=7) for _ in range(env_config['n_agents'])]

    elif policy_model == 'complex':
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation

    t = time()
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    actions = {}
    if policy_model == 'complex':
        policy_model_save = f'{policy_model}-{beta}-{gamma}-{h}-{max_price_inflation}-{max_wage_inflation}'
    if policy_model == 'gpt':
        policy_model_save = f'{policy_model}-{dialog_len}-noperception-reflection-1'
    policy_model_save = f'{policy_model_save}-{num_agents}agents-{episode_length}months'
    if not os.path.exists(f'{save_path}data/{policy_model_save}'):
        os.makedirs(f'{save_path}data/{policy_model_save}')
    if not os.path.exists(f'{save_path}figs/{policy_model_save}'):
        os.makedirs(f'{save_path}figs/{policy_model_save}')
    for epi in range(env.episode_length):
        if policy_model == 'gpt':
            actions, gpt_error, total_cost = gpt_actions(env, obs, dialog_queue, dialog4ref_queue, f'{save_path}data/{policy_model_save}/dialogs', gpt_error, total_cost, model_type=model_type)
        elif policy_model == 'complex':
            actions = complex_actions(env, obs, beta=beta, gamma=gamma, h=h)
        obs, rew, done, info = env.step(actions)
        if (epi+1) % 3 == 0:
            print(f'step {epi+1} done, cost {time()-t:.1f}s')
            if policy_model == 'gpt':
                print(f'#errors: {gpt_error}, cost ${total_cost:.1f} so far')
            t = time()
        if (epi+1) % 6 == 0 or epi+1 == env.episode_length:
            with open(f'{save_path}data/{policy_model_save}/actions_{epi+1}.pkl', 'wb') as f:
                pkl.dump(actions, f)
            with open(f'{save_path}data/{policy_model_save}/obs_{epi+1}.pkl', 'wb') as f:
                pkl.dump(obs, f)
            with open(f'{save_path}data/{policy_model_save}/env_{epi+1}.pkl', 'wb') as f:
                pkl.dump(env, f)
            if policy_model == 'gpt':
                with open(f'{save_path}data/{policy_model_save}/dialog_{epi+1}.pkl', 'wb') as f:
                    pkl.dump(dialog_queue, f)
                with open(f'{save_path}data/{policy_model_save}/dialog4ref_{epi+1}.pkl', 'wb') as f:
                    pkl.dump(dialog4ref_queue, f)
            with open(f'{save_path}data/{policy_model_save}/dense_log_{epi+1}.pkl', 'wb') as f:
                pkl.dump(env.dense_log, f)
                
    with open(f'{save_path}data/{policy_model_save}/dense_log.pkl', 'wb') as f:
        pkl.dump(env.dense_log, f)
        
    if policy_model == 'gpt':
        print(f'#gpt errors: {gpt_error}')

if __name__ == "__main__":
    fire.Fire(main)