#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

class ODENet(object):
    
    def __init__(self,F = None):
        assert callable(F);
        self.F = F;

    def aug_F(self, aug_ft, t, theta = None):
        assert issubclass(aug_ft, tf.Tensor) and aug_ft.get_shape()[0] == 4;
        # this function do the following calculation
        # d augmented f(t)/dt = augmented F(augmented f(t),t,theta)
        # inputs:
        # 1)augmented f(t) which is [f(t),a(t) = dloss/df(t), a_theta(t), a_t(t)]
        # 2)t is t
        # 3)theta is theta
        ft, dloss_dft, _, _ = tf.unstack(aug_ft);
        #get jacobian of F
        with tf.GradientTape() as g:
            g.watch([ft, t, theta]);
            F = self.F(ft,t,theta);
        dF_dft = g.gradient(F,ft); #dF/df(t)
        dF_dtheta = g.gradient(F,theta); #dF/dtheta
        dF_dt = g.gradient(F,t); #dF/dt
        #get d augmented f(t) / dt
        da_dt = -dloss_dft * dF_dft; # da(t)/dt = -a(t)^T * dF/df(t)
        datheta_dt = -dloss_dft * dF_dtheta; # da_theta(t)/dt = -a(t)^T * dF/dtheta
        dat_dt = -dloss_dft * dF_dt; # da_t(t)/dt = -a(t)^T * dF/dt
        # output: 
        # d augmented f(t) / dt which is [df(t)/dt = F, da(t)/dt, da_theta(t)/dt, da_t(t)/dt]
        return tf.stack([F, da_dt, datheta_dt, dat_dt]);
    
    def gradient(self, yt, t, theta, aT = None):
        # inputs:
        # 1)yt is an array of f(t)
        # 2)t is an array of t where the supervised value f(t) is sampled
        # 3)theta is the parameter
        # 4)aT is gradient of the loss respect to last output, dloss(f(T-1),tilde{f})/df(T-1)
        assert issubclass(type(yt), tf.Tensor) and issubclass(type(t), tf.Tensor) and issubclass(type(aT),tf.Tensor);
        assert len(t.get_shape()) == 1;
        assert yt.get_shape()[0] == t.get_shape()[0];
        assert yt.get_shape()[1:] == aT.get_shape();
        at = aT; #dloss/df(T-1) = dloss(f(T-1),tilde{f})/df(T-1)
        T = int(t.get_shape()[0]);
        sum_dloss_dt = 0;
        dloss_dtheta = tf.zeros(theta.get_shape());
        for i in range(T - 1, 0, -1):
            # -dloss / dt = -df(t) / dt * dloss / df(t) = F(f(t),t,theta) * (-dloss / df(t)) = F(f(t),t,theta) * a(t)
            # the real dloss_dt needs one more tf.math.reduce, it is not practiced to fit the dimension requirements of tf.stack
            dloss_dt = at * self.F(yt[i], t[i], theta);
            # sum_{t=t}^{T-1} dloss / dt
            sum_dloss_dt = sum_dloss_dt - dloss_dt;
            # initial condition [f(t),a(t) = dloss/df(t), a_theta(t), a_t(t)]
            # get augmented func of the previous time [f(t-1), a(t-1), a_theta(t-1), a_t(t-1)]
            aug_ft = tf.stack([yt[i], at, dloss_dtheta, sum_dloss_dt]);
            #NOTE: tensorflow doesnt support odeint with time points given in descending order
            aug_ftm1 = tf.contrib.integrate.odeint(self.aug_F, aug_ft, [t[i],t[i-1]]);
            _, atm1, dloss_dtheta, sum_dloss_dtm1 = tf.unstack(aug_ftm1[1]);
            # update a(t-1)
            at = atm1;
        return dloss_dtheta;

if __name__ == "__main__":
    tf.enable_eager_execution();
    
    def lorenz(ft, t, theta = [28.0, 10.0, 8.0/3.0]):
        x, y, z = tf.unstack(ft);
        rho, sigma, beta = tf.unstack(theta);
        dx = sigma * (y - x);
        dy = x * (rho - z) - y;
        dz = x * y - beta * z;
        return tf.stack([dx, dy, dz]);

    odenet = ODENet(lorenz);
    #time points [x_0, ... , x_{T-1}]
    t = tf.constant(np.linspace(0, 10, num = 10), dtype = tf.float32);
    #samples [f(x_0), ..., f(x_{T-1})]
    yt = tf.constant(np.random.normal(size = (10,3)), dtype = tf.float32);
    #initial loss of the last timepoint -dloss/df(x_{T-1})
    aT = tf.constant(-np.ones([3]), dtype = tf.float32); #-dloss/df(x) is 3d vector
    #initial value of trainable parameter
    theta = tf.Variable([28.0, 10.0, 8.0/3.0], dtype = tf.float32);
    #learning rate
    lr = 1e-3;
    #Newton Descend
    while True:
        dloss_dtheta = odenet.gradient(yt,t,theta,aT);
        print(dloss_dtheta);
        if tf.math.reduce(dloss_dtheta) < 1e-2: break;
        theta = theta - lr * dloss_dtheta;
