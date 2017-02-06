import numpy as np


def projected_incumbent_estimation(model, X, proj_value=1):
    projection = np.ones([X.shape[0], 1]) * proj_value
    X_projected = np.concatenate((X, projection), axis=1)

    m, _ = model.predict(X_projected)

    best = np.argmin(m)
    incumbent = X_projected[best]
    incumbent_value = m[best]

    return incumbent, incumbent_value
def projected_incumbent_optimization(model,lower,upper):
    print model
    print model.X
    def f(x,aux):
        x_proj = np.hstack([x,np.array([proj_value])])[np.newaxis,:]
        y = model.predict(x_proj)
#        print y
        return y[0][0],0.
    xmin,ymin,ierror = DIRECT.solve(f,lower,upper,maxf=1000)
    print "incumbent from posterior optimization: {} {}".format(xmin, ymin)
    if False:
        from matplotlib import pyplot as plt
        import scipy as sp
        n = 100
        x_ = sp.linspace(-1, 1, n)
        y_ = sp.linspace(-1, 1, n)
        z_ = sp.empty([n, n])
        s_ = sp.empty([n, n])
        for i in range(n):
            print '\r{}'.format(i),
            for j in range(n):
                m_, v_ = model.predict(np.array([[y_[j], x_[i],1.]]))
                z_[i, j] = m_[0]
                s_[i, j] = sp.sqrt(v_[0])
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 20))
        CS = ax[0].contour(x_, y_, z_, 20)
        ax[0].clabel(CS, inline=1, fontsize=10)
        CS = ax[1].contour(x_, y_, s_, 20)
        ax[0].axis([-1., 1., -1., 1.])
        ax[1].clabel(CS, inline=1, fontsize=10)
        plt.savefig('dbout/fab{}.png'.format(model.X.shape[0]))
        plt.close(fig)
    return np.hstack([xmin,np.array([1.])]),ymin
