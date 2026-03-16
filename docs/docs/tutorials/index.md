---
icon: material/school
---

# :material-school: Tutorials

This section presents a collection of **Jupyter Notebook** tutorials
that demonstrate how to use EasyDynamics for various tasks. These
tutorials serve as self-contained, step-by-step **guides** to help users
grasp the workflow of data analysis using EasyDynamics.

Instructions on how to run the tutorials are provided in the
[:material-cog-box: Installation & Setup](../installation-and-setup/index.md#how-to-run-tutorials)
section of the documentation.

The tutorials are organized into the following categories:

## Getting Started

We are working on expanding the list of tutorials to include advanced
concept such as interpolating `Parameter`s, sharing `Parameter`s at
multiple `Q` and analysing complex inelastic data.

- [Tutorial 1: Brownian Diffusion](tutorial1_brownian.ipynb) - Learn how
  to analyse QENS data with elastic incoherent background and Brownian
  diffusion.
- [Tutorial 2: Magnetic nanoparticles](tutorial2_nanoparticles.ipynb) -
  Learn how to do advanced QENS and INS analysis to understand the
  magnetic dynamics of nanoparticles, fitting and subsequently fixing
  background parameters and fixing some parameters to be equal when
  fitting.

## Classes and Methods

Here we go into more detail with each class that is used in the
tutorials.

- [Components](components.ipynb) – Learn how to use the EasyDynamics
  model components, which are the basic building blocks of your model.
- [Component collection](component_collection.ipynb) – Learn how to
  create a collection of components for fitting.
- [Convolution](convolution.ipynb) – Learn how to calculate the
  convolution of your resolution function with your model.
- [Detailed balance](detailed_balance.ipynb) – Learn about detailed
  balancing.
- [Diffusion model](diffusion_model.ipynb) – Learn how to create and use
  a model of diffusion.
- [Sample model](sample_model.ipynb) – Learn how to create a model of
  the scattering from your sample including model components and
  diffusion models.
- [Instrument model](instrument_model.ipynb) – Learn how to create a.
  model of your instrument including resolution and background.
- [Experiment](experiment.ipynb) - Learn how to load and bin your data.
- [Analysis](analysis.ipynb) - Learn how to fit a model to your data.
- [Analysis 1D](analysis1d.ipynb) - Learn how to fit a model to your
  data at a particular Q.
