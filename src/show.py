

import networkx as nx
import matplotlib.pyplot as plt
from stellargraph import StellarDiGraph


# 使用 NetworkX 进行可视化
pos = nx.spring_layout(nx_graph)  # 使用 spring layout 布局算法
nx.draw(nx_graph, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=20)
plt.show()
