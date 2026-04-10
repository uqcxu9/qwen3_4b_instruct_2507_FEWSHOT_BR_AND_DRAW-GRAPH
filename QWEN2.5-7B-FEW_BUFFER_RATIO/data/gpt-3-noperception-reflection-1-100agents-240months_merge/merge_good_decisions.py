import pandas as pd
import os

base_path = '/workspace/QWEN2.5_42_7b_BUFFER/data/gpt-3-noperception-reflection-1-100agents-240months_merge'

file_paths = [
    os.path.join(base_path, 'good_decisions.csv_1'),
    os.path.join(base_path, 'good_decisions.csv_2'),
    os.path.join(base_path, 'good_decisions.csv_3'),
    os.path.join(base_path, 'good_decisions.csv_4'),
    os.path.join(base_path, 'good_decisions.csv_5'),
]

dfs = []
for i, path in enumerate(file_paths, start=1):
    df = pd.read_csv(path)
    df['source_run'] = i
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

print(f'Total merged decisions: {len(df_all)}')
print('\nSource run distribution:')
print(df_all['source_run'].value_counts().sort_index())

print('\nMacro state distribution before balancing:')
print(df_all['macro_state'].value_counts(dropna=False))

print('\nEmployment distribution before balancing:')
print(df_all['work_decision'].value_counts(dropna=False))

output_path = os.path.join(base_path, 'good_decisions_merged.csv')
df_all.to_csv(output_path, index=False)

print(f'\nMerged file saved to: {output_path}')
print(pd.crosstab(df_all['macro_state'], df_all['work_decision'], normalize='index'))
print(pd.crosstab(df_all['source_run'], df_all['macro_state']))