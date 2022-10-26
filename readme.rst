Template for surrogate model generation
***************************************

It generates a surrogate model for a given function and reduces the parameters to a specified number.

``parameters`` refers to the parameters of the function "``origin_function``" the surrogate model will be built for. It is defined as a dictionary with the name of each parameter as key; the corresponding value is a tuple containing the parameter's minimum and maximum value.

| Samples are generated from the ``origin_function`` to train and to test the regressor.
| ``lhs_sample`` generates the training dataset using a `latin hypercube model <https://en.wikipedia.org/wiki/Latin_hypercube_sampling>`_
| ``random_sample`` generates the test dataset


Alternatively it is of course possible to load & split existing data into a training and test dataset.

``k`` defines the number of parameters that should be used for the surrogate model. It has to be smaller than or equal to the number of real parameters.

The best regressor is selected based on the criterion `R2 <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_ and stored in the file ``regressor.pkl``

