import matplotlib.pyplot as plt
from collections import Counter

with open('misclassified_texts/ResNet-misclassified_fakes_epochs30.txt', 'r') as f:
    paths = f.read().split()

categories = [path.split('/')[3] for path in paths if path.strip()]

counts = Counter(categories)

colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple', 'sandybrown', 'lightpink']

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(counts.keys(), counts.values(), 
              color=colors[:len(counts)], edgecolor='white', width=0.5)

for bar, count in zip(bars, counts.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{count}\n({count/sum(counts.values())*100:.1f}%)',
            ha='center', va='bottom', fontsize=11)

ax.set_title('ResNet-50 Class distribution', fontsize=13)
ax.set_xlabel('Category', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, max(counts.values()) * 1.2)

plt.tight_layout()
plt.savefig('category_distribution.png', dpi=150)
plt.show()