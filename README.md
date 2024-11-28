# WaterControl

This is a small research project to see if we can answer the following questions:

> Consider a bucket of water which we can shake/accelerate in 2 dimensions. Can we find a shaking pattern such that the surface of the water at time $T$ approximates some desired shape? And what if that shape is chosen so that light shining down through the water forms a desired image on the bottom?

We model the water with the shallow water equations, assume the acceleration is piecewise constant in time, and then perform adjoint optimization over the acceleration to minimize the loss function

$$
O = \frac{1}{2} \int \int \left| h(x,y,T) - h_\text{target}(x,y) \right|^2 \, dA
$$

This is a work in progress. Check out the example notebooks.
