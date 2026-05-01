import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../train_dataset_splits.csv')

last_col = df.columns[-2]
counts = df[last_col].value_counts().sort_index()

label_map = {0: 'Real', 1: 'Fake'}
labels = [label_map[i] for i in counts.index]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels, counts.values, color=['steelblue', 'coral'], edgecolor='white', width=0.5)

for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{count}\n({count/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontsize=11)

ax.set_title(f"Class Distribution of Real and Fake Clips", fontsize=13)
ax.set_xlabel('Class', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, counts.max() * 1.2)

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()