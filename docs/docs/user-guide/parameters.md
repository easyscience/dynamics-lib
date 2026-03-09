## Parameter Attributes

Parameters in EasyDynamics are more than just variables. They are objects
that, in addition to the name and value, also include attributes such as the
description, unit, uncertainty, minimum and maximum values, etc. All these
attributes are described in the [API Reference](../api-reference/index.md)
section. Examples of how to use these parameters in code are provided in the
[Tutorials](../tutorials/index.md) section.

The most important attribute, besides `name` and `value`, is `fixed`, which is
used to define whether the parameter is free or fixed for optimization during
the fitting process. 

Although parameters are central, EasyDynamics hides their creation and
attribute handling from the user. The user only accesses the required parameters
through the top-level objects, such as `sample_model`,
`experiment`, etc. The parameters are created and initialized automatically
when a new object is created or an existing one is loaded.

!!! warning "Important"

    Remember that parameters are accessed in code through their parent objects,
    such as `sample_model`, or `ComponentCollection`. For example, if you
    have a `ComponentCollection` and want to access the area of the first component, the syntax is:

    ```python
    component_collection.components[0].area


