import numpy, matplotlib.pyplot as plt

def CostIterationPlot(costs: list[float]):
    """
    Cost versus iterations of gradient descent, a.k.a, Learning Curve
    A plot of cost versus iterations is a useful measure of progress in gradient descent. Cost should always decrease in successful runs. 
    The change in cost is so rapid initially, it is useful to plot the initial decent on a different scale than the final descent. 
    In the plots below, note the scale of cost on the axes and the iteration step.    
    """
    # plot cost versus iteration  
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4)) # figsize = (width, height)
    ax1.plot(costs[:100])
    ax2.plot(1000 + numpy.arange(len(costs[1000:])), costs[1000:])
    ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
    ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
    ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
    plt.legend()
    plt.show()