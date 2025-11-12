import pydotplus, io, matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.tree import export_graphviz
from six import StringIO

def PlotDecisionTree(tree, features, classes, title:str):
    dot_data = StringIO()
    export_graphviz(tree, feature_names=features, out_file=dot_data,  
                    filled=True, rounded=True, proportion=True, special_characters=True,
                    impurity=False, class_names=classes, precision=2)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # Render the graph to a PNG byte string
    image = graph[0].create_png() if isinstance(graph, list) else graph.create_png()
    # Treat the DOT output as an image file in memory
    sio = io.BytesIO(image)
    img = mpimg.imread(sio)
    # Clear the current axes and plot new data
    #plt.cla()
    #plt.clf()
    # Plot the image using Matplotlib
    plt.figure(figsize=(20, 10), constrained_layout=True)
    plt.imshow(img, aspect='equal')
    plt.axis('off') # Hide axis
    plt.title(title, fontsize=22, fontweight="bold", y=1.05)
    plt.savefig(f"output/{title}.png")
    #plt.show()
