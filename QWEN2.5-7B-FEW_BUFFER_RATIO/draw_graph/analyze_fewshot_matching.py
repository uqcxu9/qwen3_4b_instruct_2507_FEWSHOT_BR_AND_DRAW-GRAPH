"""
独立分析脚本：不修改 simulate.py，从已保存的 dense_log 复现 few-shot 匹配逻辑，
并输出 (1) 每个 agent 每步被选中的 3 个 example 及字段
      (2) 当前 agent 与 top-3 的对应值与差值
      (3) work_decision / consumption_prop 分布

用法: python analyze_fewshot_matching.py [--quick]
  --quick: 仅跑前 2 个 run、前 24 步，用于快速看输出与分布
"""
import os
import sys
import pickle as pkl
import numpy as np
import pandas as pd

# 与 simulate 一致
MAX_L = 168
GOOD_DECISIONS_CSV = "/workspace/QWEN2.5_42_7b_BUFFER/data/gpt-3-noperception-reflection-1-100agents-240months_merge/good_decisions_merged.csv"
RUN_DIRS = [
    "/workspace/QWEN2.5_42_7b_BUFFER/data/gpt-3-noperception-reflection-1-100agents-240months_buffer1",
    "/workspace/QWEN2.5_42_7b_BUFFER/data/gpt-3-noperception-reflection-1-100agents-240months_buffer2",
    "/workspace/QWEN2.5_42_7b_BUFFER/data/gpt-3-noperception-reflection-1-100agents-240months_buffer11",
    # "/workspace/QWEN2.5_42_7b_BUFFER/data/gpt-3-noperception-reflection-1-100agents-240months_few1",
    # "/workspace/QWEN2.5_42_7b_BUFFER/data/gpt-3-noperception-reflection-1-100agents-240months_few2",
    # "/workspace/QWEN2.5_42_7b_BUFFER/data/gpt-3-noperception-reflection-1-100agents-240months_few3",
    # "/workspace/QWEN2.5_42_7b_BUFFER/data/gpt-3-noperception-reflection-1-100agents-240months_few4",
    # "/workspace/QWEN2.5_42_7b_BUFFER/data/gpt-3-noperception-reflection-1-100agents-240months_few5",
    "/workspace/QWEN2.5_42_7b_BUFFER/data/gpt-3-noperception-reflection-1-100agents-240months_buffer12",
    "/workspace/QWEN2.5_42_7b_BUFFER/data/gpt-3-noperception-reflection-1-100agents-240months",
]


def load_good_df():
    if not os.path.exists(GOOD_DECISIONS_CSV):
        raise FileNotFoundError(GOOD_DECISIONS_CSV)
    good_df = pd.read_csv(GOOD_DECISIONS_CSV)
    good_df["curr_dpi_safe"] = good_df["curr_dpi"].clip(lower=1e-6)
    good_df["buffer_ratio"] = (
        (good_df["curr_wealth"] + good_df["curr_dpi_safe"]) / good_df["curr_dpi_safe"]
    )
    return good_df


def _get_labor_action(actions_t, agent_id):
    last = actions_t.get(agent_id, {})
    if isinstance(last, dict):
        return int(last.get("SimpleLabor", 0))
    if isinstance(last, (list, tuple)) and len(last) >= 1:
        return int(last[0])
    return 0


def get_economic_state_from_dense_log(dense_log, timestep, num_agents):
    # 到 step t 时只有 actions[0:t]、states[0:t]，与 simulate 中 env.world.timestep==t 一致
    actions_full = dense_log.get("actions", [])
    states_full = dense_log.get("states", [])
    world = dense_log.get("world", [])
    actions = actions_full[:timestep]
    states = states_full[:timestep]
    n = len(actions)
    prices = [(world[i].get("Price", 1.0) if i < len(world) else 1.0) for i in range(n)] if world else [1.0] * n

    if timestep < 24 or len(actions) < 24:
        if len(actions) == 0:
            return "normal"
        return "normal"

    def _monthly_gdp_proxy(t):
        if t < 0 or t >= len(actions) or t >= len(states) or t >= len(prices):
            return 0.0
        supply = 0.0
        for i in range(num_agents):
            aid = str(i)
            labor = _get_labor_action(actions[t], aid)
            if labor <= 0:
                continue
            st = states[t].get(aid, {}) if isinstance(states[t], dict) else {}
            skill = float(st.get("skill", 0.0) or 0.0) if isinstance(st, dict) else 0.0
            supply += labor * MAX_L * skill
        return supply * float(prices[t])

    last12 = sum(_monthly_gdp_proxy(t) for t in range(len(actions) - 12, len(actions)))
    prev12 = sum(_monthly_gdp_proxy(t) for t in range(len(actions) - 24, len(actions) - 12))
    gdp_growth = (last12 - prev12) / (prev12 + 1e-8)
    if gdp_growth <= -0.02:
        return "recession"
    if gdp_growth >= 0.058:
        return "boom"
    return "normal"


def select_top3(
    good_df,
    curr_income,
    curr_wealth,
    curr_dpi,
    curr_buffer_ratio,
    current_macro_state,
    prev_work_decision_agent,
    timestep,
):
    """尽量复现当前 simulate.py 的 few-shot 检索逻辑"""
    curr_dpi_safe = max(curr_dpi, 1e-6)
    cold_start = (
        timestep <= 1 and
        curr_income <= 1e-6 and
        curr_wealth <= 1e-6
    )

    # 1) 先按上一期是否工作分桶
    if "prev_work_decision" in good_df.columns:
        if prev_work_decision_agent == 0:
            state_candidates = good_df[
                good_df["prev_work_decision"] == 0.0
            ].copy()
        else:
            state_candidates = good_df[
                good_df["prev_work_decision"] == 1.0
            ].copy()
    else:
        state_candidates = good_df.copy()

    # 如果桶太小，回退全库
    if len(state_candidates) < 10:
        state_candidates = good_df.copy()

    # 2) 再按 macro_state
    if "macro_state" in good_df.columns:
        macro_candidates = state_candidates[
            state_candidates["macro_state"] == current_macro_state
        ].copy()
        if len(macro_candidates) >= 5:
            state_candidates = macro_candidates

    # 3) cold-start / 常规阶段分支
    if cold_start:
        q_income = state_candidates["curr_income"].quantile(0.30)
        q_wealth = state_candidates["curr_wealth"].quantile(0.30)
        q_dpi = state_candidates["curr_dpi_safe"].quantile(0.40)

        cold_candidates = state_candidates[
            (state_candidates["curr_income"] <= max(q_income, 1.0)) &
            (state_candidates["curr_wealth"] <= max(q_wealth, 1.0)) &
            (state_candidates["curr_dpi_safe"] <= max(q_dpi, 1.0))
        ].copy()

        if len(cold_candidates) < 10:
            cold_candidates = state_candidates[
                (state_candidates["curr_income"] <= max(q_income, 1.0)) &
                (state_candidates["curr_dpi_safe"] <= max(q_dpi, 1.0))
            ].copy()

        if len(cold_candidates) < 10:
            cold_candidates = state_candidates[
                state_candidates["curr_dpi_safe"] <= max(q_dpi, 1.0)
            ].copy()

        if len(cold_candidates) >= 5:
            state_candidates = cold_candidates

    else:
        if curr_income > 1e-6:
            income_candidates = state_candidates[
                (state_candidates["curr_income"] >= curr_income * 0.5) &
                (state_candidates["curr_income"] <= curr_income * 2.0)
            ].copy()
            if len(income_candidates) >= 10:
                state_candidates = income_candidates

        dpi_candidates = state_candidates[
            (state_candidates["curr_dpi_safe"] >= curr_dpi_safe * 0.5) &
            (state_candidates["curr_dpi_safe"] <= curr_dpi_safe * 2.0)
        ].copy()
        if len(dpi_candidates) >= 10:
            state_candidates = dpi_candidates

    candidates = state_candidates.copy()
    if len(candidates) == 0:
        return pd.DataFrame()

    # 4) 计算距离
    curr_wealth_safe = max(curr_wealth, 0.0)

    candidates["buffer_distance"] = (
        np.log1p(candidates["buffer_ratio"]) - np.log1p(curr_buffer_ratio)
    ).abs()

    candidates["dpi_distance"] = (
        np.log1p(candidates["curr_dpi_safe"]) - np.log1p(curr_dpi_safe)
    ).abs()

    candidates["wealth_distance"] = (
        np.log1p(candidates["curr_wealth"].clip(lower=0.0)) -
        np.log1p(curr_wealth_safe)
    ).abs()

    # 5) cold-start / 常规阶段两套 score
    if cold_start:
        candidates["match_score"] = (
            0.10 * candidates["buffer_distance"] +
            0.45 * candidates["dpi_distance"] +
            0.45 * candidates["wealth_distance"]
        )
    else:
        candidates["match_score"] = (
            0.4 * candidates["buffer_distance"] +
            0.4 * candidates["dpi_distance"] +
            0.2 * candidates["wealth_distance"]
        )

    candidates = candidates.sort_values("match_score")
    top_decisions = candidates.head(3).copy()

    # 6) 多样性补丁：只在 top-20 里找反例
    if len(top_decisions) > 0 and top_decisions["work_decision"].nunique() == 1:
        search_pool = candidates.head(20)
        alt_pool = search_pool[
            (search_pool["work_decision"] != top_decisions["work_decision"].iloc[0]) &
            (~search_pool.index.isin(top_decisions.index))
        ]
        if len(alt_pool) > 0:
            top_decisions = pd.concat([top_decisions.iloc[:2], alt_pool.head(1)])

    top_decisions = top_decisions.sort_values("match_score")
    return top_decisions.head(3)


def load_dense_log(run_dir):
    for name in ["dense_log_240.pkl", "dense_log.pkl"]:
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pkl.load(f)
    return None


def main():
    quick = "--quick" in sys.argv
    if quick:
        run_dirs = RUN_DIRS[:2]
        max_steps = 24
        print("【quick 模式】仅前 2 个 run、前 24 步")
    else:
        run_dirs = RUN_DIRS
        max_steps = None

    good_df = load_good_df()
    print(f"Loaded good_decisions: {len(good_df)} rows")
    print()

    # 用于 (1)(2) 的明细：只对少量 (run, t, agent) 打印，避免刷屏；其余汇总
    sample_per_run = 2  # 每个 run 抽 2 个 (t, agent) 打印完整 3 example
    rows_for_1_2 = []

    all_work = []
    all_consumption_prop = []
    per_run_work = []   # (run_name, labor)
    per_run_cons = []   # (run_name, consumption_prop)

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        dense_log = load_dense_log(run_dir)
        if dense_log is None:
            print(f"Skip (no dense_log): {run_name}")
            continue

        states = dense_log["states"]
        actions = dense_log["actions"]
        periodic_tax = dense_log.get("PeriodicTax", [])
        num_agents = len([k for k in states[0] if k != "p"]) if states else 0
        T = min(len(states), len(periodic_tax), len(actions))
        if max_steps is not None:
            T = min(T, max_steps + 1)

        rng = np.random.default_rng(42)
        sampled = set()
        for t in range(1, T):
            current_macro = get_economic_state_from_dense_log(dense_log, t, num_agents)
            for agent_id in range(num_agents):
                aid = str(agent_id)
                if t - 1 >= len(states) or aid not in states[t - 1]:
                    continue
                st = states[t - 1][aid]
                pt = periodic_tax[t - 1] if t - 1 < len(periodic_tax) else {}
                pt_ag = pt.get(aid, {})
                curr_income = float(st.get("income", {}).get("Coin", 0))
                wealth = float(st.get("inventory", {}).get("Coin", 0))
                tax_paid = float(pt_ag.get("tax_paid", 0))
                lump_sum = float(pt_ag.get("lump_sum", 0))
                curr_dpi = curr_income + lump_sum - tax_paid
                curr_dpi_safe = max(curr_dpi, 1e-6)
                curr_buffer_ratio = (wealth + curr_dpi_safe) / curr_dpi_safe

                prev_action = actions[t - 1].get(aid, {}) if (t - 1) < len(actions) else {}
                if isinstance(prev_action, dict):
                    prev_work_decision_agent = int(prev_action.get("SimpleLabor", 0))
                elif isinstance(prev_action, (list, tuple)) and len(prev_action) >= 1:
                    prev_work_decision_agent = int(prev_action[0])
                else:
                    prev_work_decision_agent = 0

                top3 = select_top3(
                    good_df,
                    curr_income,
                    wealth,
                    curr_dpi,
                    curr_buffer_ratio,
                    current_macro,
                    prev_work_decision_agent,
                    t,
                )
                if top3 is None or len(top3) == 0:
                    continue

                # 实际采取的 work / consumption（从 actions[t] 取）
                act = actions[t].get(aid, {}) if t < len(actions) else {}
                if isinstance(act, dict):
                    labor = int(act.get("SimpleLabor", 0))
                    cons_idx = int(act.get("SimpleConsumption", 25))
                elif isinstance(act, (list, tuple)) and len(act) >= 2:
                    labor = int(act[0])
                    cons_idx = int(act[1])
                else:
                    labor = 0
                    cons_idx = 25
                consumption_prop = cons_idx * 0.02
                all_work.append(labor)
                all_consumption_prop.append(consumption_prop)
                per_run_work.append((run_name, labor))
                per_run_cons.append((run_name, consumption_prop))

                for ex_idx, (_, row) in enumerate(top3.iterrows()):
                    rows_for_1_2.append({
                        "run": run_name,
                        "timestep": t,
                        "agent_id": agent_id,
                        "example_rank": ex_idx + 1,
                        "agent_curr_income": curr_income,
                        "agent_curr_wealth": wealth,
                        "agent_curr_dpi": curr_dpi,
                        "agent_curr_buffer_ratio": curr_buffer_ratio,
                        "ex_curr_income": row["curr_income"],
                        "ex_curr_wealth": row["curr_wealth"],
                        "ex_curr_dpi": row["curr_dpi"],
                        "ex_buffer_ratio": row["buffer_ratio"],
                        "ex_work_decision": row["work_decision"],
                        "ex_consumption_prop": row["consumption_prop"],
                        "ex_macro_state": row.get("macro_state", ""),
                        "ex_source_run": row.get("source_run", ""),
                        "ex_timestep": row.get("timestep", ""),
                        "ex_agent_id": row.get("agent_id", ""),
                        "diff_income": row["curr_income"] - curr_income,
                        "diff_wealth": row["curr_wealth"] - wealth,
                        "diff_dpi": row["curr_dpi"] - curr_dpi,
                        "diff_buffer_ratio": row["buffer_ratio"] - curr_buffer_ratio,
                    })

                if len(sampled) < sample_per_run and (run_name, t, agent_id) not in sampled:
                    sampled.add((run_name, t, agent_id))
                    print(f"========== 示例: {run_name}  t={t}  agent={agent_id} ==========")
                    print(f"  当前 agent: curr_income={curr_income:.2f}, curr_wealth={wealth:.2f}, curr_dpi={curr_dpi:.2f}, curr_buffer_ratio={curr_buffer_ratio:.2f}")
                    print("  Top-3 examples:")
                    for i, (_, row) in enumerate(top3.iterrows(), 1):
                        print(
                            f"    Example {i}: curr_income={row['curr_income']:.2f}, curr_wealth={row['curr_wealth']:.2f}, "
                            f"curr_dpi={row['curr_dpi']:.2f}, buffer_ratio={row['buffer_ratio']:.2f}, "
                            f"work_decision={row['work_decision']}, consumption_prop={row['consumption_prop']:.2f}, "
                            f"macro_state={row.get('macro_state','')}, source_run={row.get('source_run','')}"
                        )
                        print(
                            f"      差值: income={row['curr_income']-curr_income:.2f}, wealth={row['curr_wealth']-wealth:.2f}, "
                            f"dpi={row['curr_dpi']-curr_dpi:.2f}, buffer_ratio={row['buffer_ratio']-curr_buffer_ratio:.2f}"
                        )
                    print()

    # (1)(2) 汇总表
    if rows_for_1_2:
        df = pd.DataFrame(rows_for_1_2)
        out_csv = "/workspace/QWEN2.5_42_7b_BUFFER/draw_graph/fewshot_match_detail.csv"
        df.to_csv(out_csv, index=False)
        print(f"已写入明细表: {out_csv}  (行数={len(df)})")
        print("前 5 行预览:")
        print(df.head().to_string())
        print()

        # 每个 run 的四类 MAE（top-3 example 平均误差），顺序 buffer1 buffer2 few1..few5
        run_order = sorted(df["run"].unique(), key=lambda x: (0 if "buffer" in x else 1, x))
        mae_per_run = df.groupby("run").agg(
            MAE_income=("diff_income", lambda x: np.abs(x).mean()),
            MAE_wealth=("diff_wealth", lambda x: np.abs(x).mean()),
            MAE_dpi=("diff_dpi", lambda x: np.abs(x).mean()),
            MAE_buffer_ratio=("diff_buffer_ratio", lambda x: np.abs(x).mean()),
        ).reindex(run_order)
        out_mae_run = "/workspace/QWEN2.5_42_7b_BUFFER/draw_graph/fewshot_MAE_per_run.csv"
        mae_per_run.to_csv(out_mae_run)
        print("========== 各 run 的 MAE（top-3 example 平均误差）==========")
        print(mae_per_run.to_string())
        print(f"已写入: {out_mae_run}")
        print()

        # few vs buffer 匹配误差对比表（按组汇总）
        df["run_type"] = df["run"].apply(lambda r: "buffer" if "buffer" in r else "few")
        mae = df.groupby("run_type").agg(
            MAE_income=("diff_income", lambda x: np.abs(x).mean()),
            MAE_wealth=("diff_wealth", lambda x: np.abs(x).mean()),
            MAE_dpi=("diff_dpi", lambda x: np.abs(x).mean()),
            MAE_buffer_ratio=("diff_buffer_ratio", lambda x: np.abs(x).mean()),
        ).T
        mae = mae[["buffer", "few"]] if "few" in mae.columns and "buffer" in mae.columns else mae
        out_compare = "/workspace/QWEN2.5_42_7b_BUFFER/draw_graph/fewshot_match_error_buffer_vs_few.csv"
        mae.to_csv(out_compare)
        print("========== few vs buffer 匹配误差对比表 (MAE = 均绝对差值) ==========")
        print(mae.to_string())
        print(f"已写入: {out_compare}")
        print()

        # example 重复率表：每 run 总槽位数、唯一 example 数、重复率(1 - 唯一/总)
        def unique_example_key(grp):
            return grp["ex_source_run"].astype(str) + "_" + grp["ex_timestep"].astype(str) + "_" + grp["ex_agent_id"].astype(str)

        rep_rows = []
        for run in run_order:
            sub = df[df["run"] == run]
            total_slots = len(sub)
            sub = sub.copy()
            sub["ex_key"] = unique_example_key(sub)
            unique_count = sub["ex_key"].nunique()
            repeat_rate = 1.0 - (unique_count / total_slots) if total_slots else 0.0
            rep_rows.append({
                "run": run,
                "总槽位数": total_slots,
                "唯一example数": unique_count,
                "重复率": round(repeat_rate, 4),
            })
        rep_df = pd.DataFrame(rep_rows)
        out_rep = "/workspace/QWEN2.5_42_7b_BUFFER/draw_graph/fewshot_example_repeat_rate.csv"
        rep_df.to_csv(out_rep, index=False)
        print("========== example 重复率表 ==========")
        print(rep_df.to_string(index=False))
        print(f"已写入: {out_rep}")
        print()

    # (3) work_decision / consumption_prop 分布（整体 + 按 run 看有没有变）
    print("========== (3) work_decision / consumption_prop 分布 ==========")
    if all_work:
        work_arr = np.array(all_work)
        cons_arr = np.array(all_consumption_prop)
        print("【整体】work_decision (0=失业 1=就业):")
        print(f"  mean={work_arr.mean():.4f}, 0 占比={(work_arr==0).mean():.4f}, 1 占比={(work_arr==1).mean():.4f}")
        print("【整体】consumption_prop:")
        print(f"  mean={cons_arr.mean():.4f}, std={cons_arr.std():.4f}, min={cons_arr.min():.4f}, max={cons_arr.max():.4f}")
        for q in [0.25, 0.5, 0.75]:
            print(f"  quantile {q}={np.quantile(cons_arr, q):.4f}")
        if per_run_work:
            df_work = pd.DataFrame(per_run_work, columns=["run", "work"])
            df_cons = pd.DataFrame(per_run_cons, columns=["run", "consumption_prop"])
            print("\n【按 run 分布】work_decision 就业率(1占比):")
            for run in df_work["run"].unique():
                w = df_work[df_work["run"] == run]["work"]
                print(f"  {run}: mean={w.mean():.4f}, 0占比={(w==0).mean():.4f}, 1占比={(w==1).mean():.4f}, n={len(w)}")
            print("\n【按 run 分布】consumption_prop:")
            for run in df_cons["run"].unique():
                c = df_cons[df_cons["run"] == run]["consumption_prop"]
                print(f"  {run}: mean={c.mean():.4f}, std={c.std():.4f}, n={len(c)}")
    else:
        print("无数据（未找到 dense_log 或 PeriodicTax）")
    print("Done.")


if __name__ == "__main__":
    main()
2002     7.49%         [-0.36%, 3.90%]   Above upper bound

2003    10.77%         [-0.36%, 3.90%]   Above upper bound

2004     3.60%         [-0.36%, 3.90%]   Satisfied

2005     3.53%         [-0.36%, 3.90%]   Satisfied

2006     2.28%         [-0.36%, 3.90%]   Satisfied

2007     1.76%         [-0.36%, 3.90%]   Satisfied

2008     1.93%         [-0.36%, 3.90%]   Satisfied

2009     1.77%         [-0.36%, 3.90%]   Satisfied

2010     0.58%         [-0.36%, 3.90%]   Satisfied

2011     2.14%         [-0.36%, 3.90%]   Satisfied

2012     1.62%         [-0.36%, 3.90%]   Satisfied

2013     1.38%         [-0.36%, 3.90%]   Satisfied

2014    -0.05%         [-0.36%, 3.90%]   Satisfied

2015     0.37%         [-0.36%, 3.90%]   Satisfied

2016    -0.25%         [-0.36%, 3.90%]   Satisfied

2017     1.20%         [-0.36%, 3.90%]   Satisfied

2018     1.09%         [-0.36%, 3.90%]   Satisfied

2019    -2.82%         [-0.36%, 3.90%]   Below lower bound

2020    -0.77%         [-0.36%, 3.90%]   Below lower bound

