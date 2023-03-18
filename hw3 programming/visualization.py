from matplotlib import pyplot as plt
import matplotlib as mpl

# recommended color for different digits
color_mapping = {0:'red',1:'green',2:'blue',3:'yellow',4:'magenta',5:'orangered',
                6:'cyan',7:'purple',8:'gold',9:'pink'}

def plot2d(data,label,split='train'):
    # 2d scatter plot of the hidden features
    
    colors  = color_mapping.values()
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    
    plt.title("2D plot for " + split)
    plt.scatter(
        data[:, 0], data[:, 1], c=label, cmap=mpl.colors.ListedColormap(colors))
    plt.xlabel("Hidden 1")
    plt.ylabel("Hidden 2")
    # plt.figure(figsize=(5,1))
    plt.show()
    
    # figname =
    fig.savefig("2Dplot_" + split + '.png', dpi=100)
    pass

def plot3d(data,label,split='train'):
    # 3d scatter plot of the hidden features
    colors  = color_mapping.values()
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(data[:, 0], data[:, 1],  data[:, 2], c=label, cmap=mpl.colors.ListedColormap(colors), marker='o')
    
    ax.set_title("3D plot for " + split)
    ax.set_xlabel('Hidden 1')
    ax.set_ylabel('Hidden 2')
    ax.set_zlabel('Hidden 3')
    plt.show()
    
    # figname =
    fig.savefig("3Dplot_" + split + '.png', dpi=100)
    pass
