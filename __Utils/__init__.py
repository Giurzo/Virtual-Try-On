if __name__ == "__main__":
    import numpy as np

    X, Y = np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
    stack = np.stack([X,Y,np.ones_like(Y)],2)

    points = stack[:,:,None,:]

    import matplotlib.pyplot as plt
    zs = np.sqrt(X**2 + Y**2)
    h = plt.contourf(X, Y, zs)
    plt.axis('scaled')
    plt.colorbar()
    plt.show()

    H = np.array(
        [[1,0,0],
        [0.2,1,0],
        [0.1,0.2,1]]
    )

    H = np.array(
        [[0.32903393332201, -1.244138808862929, 536.4769088231476],
        [0.6969763913334046, -0.08935909072571542, -80.34068504082403],
        [0.00040511729592961, -0.001079740100565013, 0.9999999999999999]])
    T = points @ H

    X = T[:,:,0,0]
    Y = T[:,:,0,1]
    print(X.shape)
    print(Y.shape)
    import matplotlib.pyplot as plt
    #zs = np.sqrt(X**2 + Y**2)
    h = plt.contourf(X, Y, zs)
    plt.axis('scaled')
    plt.colorbar()
    plt.show()