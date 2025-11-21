import pydotplus, io, matplotlib.pyplot as plt, matplotlib.image as mpimg, networkx as nx
from PIL import Image
from sklearn.tree import export_graphviz
from six import StringIO
from matplotlib import get_configdir
from networkx.drawing.nx_pydot import graphviz_layout
print(f"configdir: {get_configdir()}")
plt.style.reload_library()
#print(f"style.available: {plt.style.available}")
#plt.style.use('deeplearning.mplstyle')

def generate_node_image(node_indices):
    image_paths = ["images/%d.png" % idx for idx in node_indices]
    images = [Image.open(x) for x in image_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    
    new_im = new_im.resize((int(total_width*len(node_indices)/10), int(max_height*len(node_indices)/10)))
    
    return new_im

def generate_split_viz(node_indices, left_indices, right_indices, feature):
    G=nx.DiGraph()
    indices_list = [node_indices, left_indices, right_indices]

    for idx, indices in enumerate(indices_list):
        G.add_node(idx,image= generate_node_image(indices))

    G.add_edge(0,1)
    G.add_edge(0,2)
    pos = graphviz_layout(G, prog="dot")
    fig = plt.figure(figsize=(14, 10))
    ax = plt.subplot(111)  # (nrows, ncols, index): 1 rows, 1 columns, index
    ax.set_aspect('equal')
    nx.draw_networkx_edges(G,pos,ax=ax, arrows=True, arrowsize=40)
    
    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform
    feature_name = ["Ear Shape", "Face Shape", "Whiskers"][feature]
    ax_name = ["Splitting on %s" % feature_name , "Left: %s = 1" % feature_name, "Right: %s = 0" % feature_name]
    for idx, n in enumerate(G):
        xx,yy=trans(pos[n]) # figure coordinates
        xa,ya=trans2((xx,yy)) # axes coordinates
        piesize = len(indices_list[idx])/9
        p2=piesize/2.0
        a = plt.axes([xa-p2,ya-p2, piesize, piesize])
        a.set_aspect('equal')
        a.imshow(G.nodes[n]['image'])
        a.axis('off')
        a.set_title(ax_name[idx])
    ax.axis('off')
    plt.show()

def generate_tree_viz(root_indices, y, tree):
    G=nx.DiGraph()
    G.add_node(0,image= generate_node_image(root_indices))
    idx = 1
    root = 0
    num_images = [len(root_indices)]
    feature_name = ["Ear Shape", "Face Shape", "Whiskers"]
    y_name = ["Non Cat","Cat"]
    decision_names = []
    leaf_names = []

    for i, level in enumerate(tree):
        indices_list = level[:2]
        for indices in indices_list:
            G.add_node(idx,image= generate_node_image(indices))
            G.add_edge(root, idx)
            # For visualization
            num_images.append(len(indices))
            idx += 1
            if i > 0:
                leaf_names.append("Leaf node: %s" % y_name[max(y[indices])])
        decision_names.append("Split on: %s" % feature_name[level[2]])
        root += 1
    
    node_names = decision_names + leaf_names
    pos = graphviz_layout(G, prog="dot")
    fig = plt.figure(figsize=(14, 10))
    ax = plt.subplot(111)  # (nrows, ncols, index): 1 rows, 1 columns, index
    ax.set_aspect('equal')
    nx.draw_networkx_edges(G,pos,ax=ax, arrows=True, arrowsize=40)
    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform

    for idx, n in enumerate(G):
        xx,yy=trans(pos[n]) # figure coordinates
        xa,ya=trans2((xx,yy)) # axes coordinates
        piesize = num_images[idx]/25
        p2=piesize/2.0
        a = plt.axes([xa-p2,ya-p2, piesize, piesize])
        a.set_aspect('equal')
        a.imshow(G.nodes[n]['image'])
        a.axis('off')
        try:
            a.set_title(node_names[idx], y=-0.8, fontsize=13, loc="left")
        except:
            pass
    ax.axis('off')
    plt.show()

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
    # Plot the image using Matplotlib
    plt.figure(figsize=(20, 10), constrained_layout=True)
    plt.imshow(img, aspect='equal')
    plt.axis('off') # Hide axis
    plt.title(title, fontsize=22, fontweight="bold", y=1.05)
    plt.savefig(f"output/{title}.png")
    #plt.show()
