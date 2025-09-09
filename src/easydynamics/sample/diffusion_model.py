import warnings
from typing import Dict, List, Union, Tuple

import numpy as np

from easyscience.variable import Parameter
from easyscience.base_classes import ObjBase

from easydynamics.utils import detailed_balance_factor
from .components import ModelComponent



class DiffusionModel(ObjBase):
    """
    Base class for constructing diffusion models.
    """
    def __init__(self, name="MyDiffusionModel"):
        """
        Initialize a new DiffusionModel.

        Parameters
        ----------
        name : str
            Name of the diffusion model.
        """
                
        self.components: Dict[str, ModelComponent] = {}
        super().__init__(name=name)



    def __repr__(self):
        """
        Return a string representation of the DiffusionModel.

        Returns
        -------
        str
        """

        return f"DiffusionModel(name={self.name}, parameters={self.get_parameters()})"
    



    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the model at given x values
        """      

        return NotImplementedError


    def get_parameters(self) -> List[Parameter]:
        """
        Return all parameters from the model, including temperature.

        Returns
        -------
        List[Parameter]
        """
        if isinstance(self._temperature, Parameter):
            params = [self._temperature]
        else:
            params = []
        for comp in self.components.values():
            params.extend(comp.get_parameters())
        return params
    
    def get_fit_parameters(self):
        """
        Get all fit parameters, filtering out fixed parameters.

        Returns:
            List[Parameter]: A list of unfixed fit parameters.
        """
        return [param for param in self.get_parameters() if not getattr(param, 'fixed', False)]
    
    def fix_all_parameters(self):
        """
        Fix all unfixed parameters in the model.
        """
        for param in self.get_parameters():
            param.fixed = True

    def free_all_parameters(self):
        """
        Free all fixed parameters in the model.
        """
        for param in self.get_parameters():
            param.fixed = False
    
    def copy(self) -> "DiffusionModel":
        """
        Create a deep copy of the DiffusionModel instance.

        Returns
        -------
        DiffusionModel
        """
        return NotImplementedError


class BrownianTranslationalDiffusion(DiffusionModel):
    """ Lorentzian model with half width half maximum equal to :math:`Dq^2`
    """
     
    def __init__ (self, name="BrownianTranslationalDiffusion", diffusion_coefficient=1e-10,scale=1.0):
         """
         Initialize a new BrownianTranslationalDiffusion model.

         Parameters
         ----------
         name : str
             Name of the diffusion model.
         diffusion_coefficient : float
             Diffusion coefficient .
         """
         super().__init__(name=name)
         self.diffusion_coefficient = Parameter(name="diffusion_coefficient", value=diffusion_coefficient, unit='m^2/s', fixed=False)
         self.scale = Parameter(name="scale", value=scale, unit='', fixed=False)
         self._width=None
         self._EISF=None
         self._QISF=None

    def calculate_width(self, q: np.ndarray) -> np.ndarray:
        """
        Calculate the half-width at half-maximum (HWHM) for the Brownian translational diffusion model.

        Parameters
        ----------
        q : np.ndarray
            Scattering vector 

        Returns
        -------
        np.ndarray
            HWHM values.
        """
        D = self.diffusion_coefficient.value  #TODO: handle units properly
        self._width = D * q**2
        return self._width

    def calculate_EISF(self,q: np.ndarray) -> np.ndarray:
        """
        Calculate the Elastic Incoherent Structure Factor (EISF) for the Brownian translational diffusion model.

        Parameters
        ----------
        q : np.ndarray
            Scattering vector 

        Returns
        -------
        np.ndarray
            EISF values (dimensionless).
        """
        self._EISF = np.zeros_like(q)
        return self._EISF
    
    def calculate_QISF(self,q: np.ndarray) -> np.ndarray:
        """
        Calculate the Quasi-Elastic Incoherent Structure Factor (QISF) for the Brownian translational diffusion model.

        Parameters
        ----------
        q : np.ndarray
            Scattering vector 

        Returns
        -------
        np.ndarray
            QISF values (dimensionless).
        """

        self._QISF = np.ones_like(q)
        return self._QISF

