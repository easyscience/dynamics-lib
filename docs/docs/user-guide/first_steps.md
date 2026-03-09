# First Steps

This section introduces the basic usage of the EasyDynamics Python API.
You'll learn how to import the package and use core classes and utility
functions,

## Importing EasyDynamics

### Importing the entire package

To start using EasyDynamics, first import the package in your Python
script or Jupyter Notebook. This can be done with the following command:

```python
import easydynamics
```

Alternatively, you can import it with an alias to avoid naming conflicts
and for convenience:

```python
import easydynamics as ed
```

The latter syntax allows you to access all the modules and classes
within the package using the `ed` prefix. For example, you can create an
`Analysis` instance like this:

```python
analysis = ed.Analysis()
```

### Importing specific parts

Alternatively, you can import specific classes or methods from the
package. For example, you can import the `Analysis`, `SampleModel`,
`Experiment` and `Gaussian` classes like this:

```python
from easydynamics.Analysis import Analysis
from easydynamics.Experiment import Experiment
from easydynamics.SampleModel import SampleModel, Gaussian
```

This enables you to use these classes and methods directly without the
package prefix. This is especially useful when you're using only a few
components and want to keep your code clean and concise. In this case,
you can create a `Analysis` instance like this:

```python
analysis = Analysis()
```
