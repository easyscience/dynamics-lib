The goal of EasyDynamics is to make it easy to fit QENS and powder INS data.

The fit is carried out by an `Analysis` object, which contains an `Experiment` that contains the data, a `SampleModel` that describes the scattering from the sample, and an `InstrumentModel` that describes the instrument. Within the `InstrumentModel` lives the `ResolutionModel`, which describes the instrument resolution, a `BackgroundModel`, which describes the background scattering and an offset in energy to handle small misalignments of the instrument. 
The `Analysis` object keeps track of all of this and automatically calculates the convolution of the resolution and the sample model. All objects are unit-aware and automatically handle conversion, e.g. between $\mu$eV and meV. The `Analysis` object furthermore contains methods to visualize the data, model and fit parameters.

The `SampleModel`, `BackgroundModel` and `InstrumentModel` contains a list of `ComponentCollection`s, which is essentially list of `ModelComponents`. A `ModelComponent` can be any of `Gaussian`, `Lorentzian`, `Voigt` (the convolution of a `Gaussian` and `Lorentzian`), `DeltaFunction`, `DampedHarmonicOscillator` and `Polynomium`.

Each `ModelComponent` has a number of `Parameter`s describing it. The `Gaussian`, for example, has `area`, `center` and `width`. Each of these `Parameter`s has a number of properties and methods that can be accessed as needed. The most important are, in no particular order: 

- `value`: The current value of the `Parameter` - gets updated when fitting.
- `variance`: The variance of the `Parameter` - gets updated when fitting.
- `error`: The square root of the `variance` - also gets updated when fitting.
- `min`: the minimum value of the `Parameter` when fitting it.
- `max`: the maximum value of the `Parameter` when fitting it.
- `fixed`: If True, the `Parameter` is not fitted.
- `unit`: The unit of the `Parameter`.
- `convert_unit`: Change the unit of the `Parameter.

So, for example, one can create a `Gaussian` and change its `Parameter`s like this:
```
gaussian = Gaussian()
gaussian.area=2.0
gaussian.area.fixed=True
gaussian.width.fixed=False
gaussian.width.min=0.5
```
And so on.

It is furthermore possible to make a `Parameter` depend on another `Parameter`. We refer to the examples for this.