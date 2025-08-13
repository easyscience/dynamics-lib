from easyscience.job.job import JobBase

from easydynamics.sample import SampleModel
from easydynamics.experiment import Experiment
from easydynamics.analysis import Analysis
from easydynamics.experiment.data import Data

import scipp as sc
import plopp as pp

from collections import defaultdict, Counter


import math
# import scipp as sc
# from collections import defaultdict, Counter


class Job(JobBase):
    def __init__(self, name: str, interface=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.name = name
        self._theory = None
        self._resolution_model = None
        self._background_model = None
        self._experiment = None
        self._analysis = []
        self._summary = None
        self._info = None
        self._fit_parameters = None


    def set_theory(self, theory):
        self._theory = theory

    def set_experiment(self, experiment):
        self._experiment = experiment   


    def set_background_model(self, background:SampleModel):
        """ Set the model for the background.
        Args:
            background (SampleModel): The background model.
        """
        if not isinstance(background, SampleModel):
            raise TypeError("Background model must be an instance of SampleModel.")
        self._background_model = background

    def set_resolution_model(self, resolution:SampleModel):
        """        Set the resolution model for the experiment. The resolution will be normalised to have area 1.
        Args:
            resolution (SampleModel): The resolution model to be used in the experiment.
        """
        # TODO: allow resolution to be DataArray or SampleModel

        if resolution is not None and not isinstance(resolution, SampleModel):
            raise TypeError("Resolution model must be an instance of SampleModel.")
        self._resolution_model = resolution

        if self._resolution_model is not None:
            self.normalize_resolution()

    def normalize_resolution(self):
        """ Normalize the resolution model to have an area of 1.
        """
        self._resolution_model.normalize_area()        

    def set_analysis(self, analysis):
        self._analysis.append(analysis)
        if self._experiment is not None:
            self._analysis[-1].set_experiment(self._experiment)
        if self._theory is not None:
            self._analysis[-1].set_theory(self._theory)

    def fit(self):
        if self._analysis is None:
            raise RuntimeError("Analysis is not set in Job.")

        for i in range(len(self._analysis)):
            self._analysis[i].fit()
        # return self._analysis.fit()

        self._fit_parameters=self.get_parameters_as_data_group()

    def generate_analysis_for_cuts(self):
        for i in range(self._experiment._data.data.sizes['Q']):
            this_analysis=Analysis()
            this_analysis.set_theory(self._theory.copy())

            if self._background_model is not None:
                this_analysis.set_background_model(self._background_model.copy())
            if self._resolution_model is not None:
                this_analysis.set_resolution_model(self._resolution_model.copy())

            this_experiment=Experiment()
            this_data=Data()
            this_data.append(self._experiment._data.data['Q',i])
            this_experiment.set_data(this_data)

            this_analysis.set_experiment(this_experiment)
            self._analysis.append(this_analysis)

    def plot_data_and_model(self,intensity_min=0.0, intensity_max=0.06,
                            energy_min=-0.02, energy_max=0.02):

        model = sc.zeros_like(self._experiment._data.data)

        for i in range(len(self._analysis)):
            model['Q',i].values = self._analysis[i].calculate_theory(model['Q',i].coords['energy'].values)


        data_and_fit = sc.DataGroup({'Data': self._experiment._data.data,
                                    'Fit': model})

        energy_min = energy_min * sc.Unit('meV')
        energy_max = energy_max * sc.Unit('meV')
        plot=pp.slicer(data_and_fit['energy',energy_min:energy_max],
                vmin=intensity_min,vmax=intensity_max,
                    keep=['energy'],
            linestyle=         {'Data': 'none',    'Fit': '-'},
            marker=            {'Data': 'o',       'Fit':'none'},
            markerfacecolor=   {'Data': 'none',    'Fit':'red'},
            color=             {'Data': 'black',   'Fit':'red'})

        return plot
    
    def plot_fit_parameters(self, parameter_name):
        """
        Plot the fit parameters of the analysis.

        Parameters
        ----------
        parameter_name : str, optional
            If provided, only plots the specified parameter.
            If None, plots all parameters.

        Returns
        -------
        pp.Plot
            The plot of the fit parameters.
        """
        if self._fit_parameters is None:
            raise RuntimeError("Fit parameters are not available. Run fit() first.")

        if parameter_name is not None:
            if parameter_name not in self._fit_parameters:
                raise KeyError(f"Parameter '{parameter_name}' not found in fit parameters. Available parameters: {list(self._fit_parameters.keys())}")
            return pp.plot(self._fit_parameters[parameter_name])




    def use_fit_as_resolution(self,job):
        """
        Use the fit from a Job as the resolution model.
        Args:
            job (Job): The Job containing the fit to be used as resolution.
        """
        if not isinstance(job, Job):
            raise TypeError("Job must be an instance of Job.")
        
        if job._analysis is None or len(job._analysis) == 0:
            raise RuntimeError("No analysis found in the provided job.")

        for i in range(len(self._analysis)):
            self._analysis[i].set_resolution_model(job._analysis[i]._theory.copy())
            self._analysis[i].fix_resolution_parameters()

    @property
    def analysis(self):
        return self._analysis
    
    def calculate_theory(self, x):
        return self._analysis.calculate_theory(x,_experiment=self._experiment, theory=self._theory)
    
    def experiment(self):
        return self._experiment
    
    def theoretical_model(self):
        return self._theory
    
    def get_fit_parameters(self):
        return self._analysis.get_fit_parameters()
    
    def get_parameters(self):
        return self._analysis.get_parameters()




    def get_parameters_as_data_group(self):
        N = len(self._analysis)
        q_coord = self._experiment._data.data.coords.get('Q', sc.arange('Q', N))

        # Inspect the first analysis to detect duplicate names
        first_params = self._analysis[0].get_parameters()
        name_counts = Counter(p.name for p in first_params)
        def key_for(name, idx):
            return name if name_counts[name] == 1 else f'{name}[{idx}]'

        # Template/specs
        template_seen = defaultdict(int)
        param_specs = []
        for p in first_params:
            occ = template_seen[p.name]
            template_seen[p.name] += 1
            param_specs.append((key_for(p.name, occ), p.name, occ))

        # Storage
        store = {}
        for key, _, _ in param_specs:
            store[key] = {'values': [float('nan')]*N,
                        'vars':   [None]*N,
                        'unit':   None}

        # Fill
        for i, ana in enumerate(self._analysis):
            seen = defaultdict(int)
            for p in ana.get_parameters():
                occ = seen[p.name]
                seen[p.name] += 1

                if p.name not in name_counts:
                    name_counts[p.name] = 1
                    if p.name not in store:
                        store[p.name] = {'values': [float('nan')]*N,
                                        'vars':   [None]*N,
                                        'unit':   None}

                key = key_for(p.name, occ) if name_counts[p.name] > 1 else p.name
                if key not in store:  # late duplicate discovery
                    key = f'{p.name}[{occ}]'
                    if key not in store:
                        store[key] = {'values': [float('nan')]*N,
                                    'vars':   [None]*N,
                                    'unit':   None}

                store[key]['values'][i] = p.value
                err = getattr(p, 'error', None)
                if err is not None:
                    store[key]['vars'][i] = err**2
                u = getattr(p, 'unit', None)
                if store[key]['unit'] is None and u is not None:
                    try:
                        store[key]['unit'] = sc.Unit(str(u))
                    except Exception:
                        store[key]['unit'] = None

        # Build DataGroup (coords go on DataArray, not sc.array)
        dg = {}
        for key, buf in store.items():
            include_vars = all(v is not None for v in buf['vars'])  # only include if none are missing
            data_kwargs = {}
            if buf['unit'] is not None:
                data_kwargs['unit'] = buf['unit']
            if include_vars:
                data_kwargs['variances'] = buf['vars']

            data = sc.array(dims=['Q'], values=buf['values'], **data_kwargs)
            da = sc.DataArray(data=data, coords={'Q': q_coord})
            dg[key] = da

        return sc.DataGroup(dg)


    # def get_parameters_as_data_group_with_bounds(self):
    #     """
    #     Returns a Scipp DataGroup:
    #     {
    #         'Gaussianarea': DataGroup({
    #             'value': DataArray(values[..], variances[..], coords={'Q': ..}),
    #             'min':   DataArray(values[..], coords={'Q': ..}),
    #             'max':   DataArray(values[..], coords={'Q': ..}),
    #             'fixed': DataArray(bool values[..], coords={'Q': ..}),
    #         }),
    #         'Lorentzianwidth': DataGroup({...}),
    #         ...
    #     }
    #     Duplicates (same name appearing multiple times per analysis) get keys like 'name[0]', 'name[1]'.
    #     """
    #     N = len(self._analysis)
    #     q_coord = self._experiment._data.data.coords.get('Q', sc.arange('Q', N))

    #     # --- Helpers -------------------------------------------------------------
    #     def _unit_to_sc(u):
    #         if u is None:
    #             return None
    #         try:
    #             return sc.Unit(str(u))
    #         except Exception:
    #             return None

    #     def _bounds_of(p):
    #         # Try common attribute names, then .bounds if present
    #         low = getattr(p, 'min',  getattr(p, 'minimum',  None))
    #         high = getattr(p, 'max',  getattr(p, 'maximum',  None))
    #         b = getattr(p, 'bounds', None)
    #         if (low is None or high is None) and b is not None and len(b) == 2:
    #             low = b[0] if low  is None else low
    #             high = b[1] if high is None else high
    #         return low, high

    #     def _fixed_of(p):
    #         return bool(getattr(p, 'fixed', False))

    #     # --- First pass: detect duplicates on first analysis ---------------------
    #     first_params = self._analysis[0].get_parameters()
    #     name_counts = Counter(p.name for p in first_params)

    #     def key_for(name, occ_idx):
    #         return name if name_counts[name] == 1 else f'{name}[{occ_idx}]'

    #     # Template/specs from first analysis
    #     template_seen = defaultdict(int)
    #     specs = []  # list of (key, base_name, occ_idx)
    #     for p in first_params:
    #         occ = template_seen[p.name]
    #         template_seen[p.name] += 1
    #         specs.append((key_for(p.name, occ), p.name, occ))

    #     # Storage per parameter key
    #     store = {}
    #     for key, _, _ in specs:
    #         store[key] = {
    #             'values': [math.nan]*N,
    #             'vars':   [math.nan]*N,     # NaN for missing errors
    #             'min':    [math.nan]*N,
    #             'max':    [math.nan]*N,
    #             'fixed':  [False]*N,
    #             'unit':   None
    #         }

    #     # --- Fill from all analyses ---------------------------------------------
    #     for i, ana in enumerate(self._analysis):
    #         seen = defaultdict(int)
    #         for p in ana.get_parameters():
    #             occ = seen[p.name]
    #             seen[p.name] += 1

    #             # If a new name appears later, register it
    #             if p.name not in name_counts:
    #                 name_counts[p.name] = 1
    #                 k = p.name
    #                 if k not in store:
    #                     store[k] = {
    #                         'values': [math.nan]*N, 'vars': [math.nan]*N,
    #                         'min': [math.nan]*N, 'max': [math.nan]*N,
    #                         'fixed': [False]*N, 'unit': None
    #                     }

    #             # Choose key (suffix if duplicated)
    #             k = key_for(p.name, occ) if name_counts[p.name] > 1 else p.name
    #             if k not in store:  # late duplicate discovery
    #                 k = f'{p.name}[{occ}]'
    #                 if k not in store:
    #                     store[k] = {
    #                         'values': [math.nan]*N, 'vars': [math.nan]*N,
    #                         'min': [math.nan]*N, 'max': [math.nan]*N,
    #                         'fixed': [False]*N, 'unit': None
    #                     }

    #             # Value & variance (error^2) — keep NaN if missing
    #             store[k]['values'][i] = getattr(p, 'value', math.nan)
    #             err = getattr(p, 'error', None)
    #             store[k]['vars'][i] = (err**2) if (err is not None) else math.nan

    #             # Bounds
    #             low, high = _bounds_of(p)
    #             store[k]['min'][i] = low if low is not None else math.nan
    #             store[k]['max'][i] = high if high is not None else math.nan

    #             # Fixed
    #             store[k]['fixed'][i] = _fixed_of(p)

    #             # Unit (first non-None wins)
    #             if store[k]['unit'] is None:
    #                 store[k]['unit'] = _unit_to_sc(getattr(p, 'unit', None))

    #     # --- Build the nested DataGroup -----------------------------------------
    #     out = {}
    #     for key, buf in store.items():
    #         u = buf['unit']
    #         # value with variances (NaNs allowed)
    #         val = sc.array(dims=['Q'], values=buf['values'],
    #                     variances=buf['vars'], unit=u) if u is not None \
    #             else sc.array(dims=['Q'], values=buf['values'],
    #                             variances=buf['vars'])
    #         da_value = sc.DataArray(data=val, coords={'Q': q_coord})

    #         # min/max with same unit, fixed is boolean
    #         if u is not None:
    #             da_min = sc.DataArray(sc.array(dims=['Q'], values=buf['min'], unit=u), coords={'Q': q_coord})
    #             da_max = sc.DataArray(sc.array(dims=['Q'], values=buf['max'], unit=u), coords={'Q': q_coord})
    #         else:
    #             da_min = sc.DataArray(sc.array(dims=['Q'], values=buf['min']), coords={'Q': q_coord})
    #             da_max = sc.DataArray(sc.array(dims=['Q'], values=buf['max']), coords={'Q': q_coord})

    #         da_fixed = sc.DataArray(sc.array(dims=['Q'], values=buf['fixed'], dtype='bool'), coords={'Q': q_coord})

    #         out[key] = sc.DataGroup({
    #             'value': da_value,
    #             'min':   da_min,
    #             'max':   da_max,
    #             'fixed': da_fixed,
    #         })

    #     return sc.DataGroup(out)


    # def get_parameters_as_data_group_with_T(self):
    #     data = self._experiment._data.data

    #     # ---- figure out dims/coords ----
    #     q_dim = 'Q' if 'Q' in data.dims else next((d for d in data.dims if d.lower()=='q'), 'Q')
    #     q_size = data.sizes[q_dim] if q_dim in data.dims else len(self._analysis)
    #     q_coord = data.coords.get(q_dim, sc.arange(q_dim, q_size))

    #     # temperature dimension (optional)
    #     temp_dim = 'Temperature' if 'Temperature' in data.dims else ('T' if 'T' in data.dims else None)
    #     if temp_dim:
    #         t_size  = data.sizes[temp_dim]
    #         t_coord = data.coords[temp_dim]
    #     else:
    #         t_size, t_coord = 1, None  # no temp dim

    #     # ---- map (t,q) -> analysis object ----
    #     # Accept either flat list (len==t_size*q_size) or nested list [t][q]
    #     def ana_at(t, q):
    #         if temp_dim:
    #             if isinstance(self._analysis, (list, tuple)):
    #                 if len(self._analysis) == t_size * q_size:
    #                     return self._analysis[t * q_size + q]
    #                 elif len(self._analysis) == t_size and isinstance(self._analysis[0], (list, tuple)):
    #                     return self._analysis[t][q]
    #         # no temperature dim: analysis indexed by q
    #         return self._analysis[q]

    #     # ---- duplicate-name handling based on first cell ----
    #     first_params = ana_at(0, 0).get_parameters()
    #     name_counts = Counter(p.name for p in first_params)
    #     def key_for(name, occ_idx):
    #         return name if name_counts[name] == 1 else f'{name}[{occ_idx}]'

    #     template_seen = defaultdict(int)
    #     specs = []  # (key, base_name, occ_idx)
    #     for p in first_params:
    #         occ = template_seen[p.name]
    #         template_seen[p.name] += 1
    #         specs.append((key_for(p.name, occ), p.name, occ))

    #     # ---- storage buffers ----
    #     if temp_dim:
    #         shape = (t_size, q_size)
    #         dims_out = [temp_dim, q_dim]
    #         coords_out = {q_dim: q_coord, temp_dim: t_coord}
    #     else:
    #         shape = (q_size,)
    #         dims_out = [q_dim]
    #         coords_out = {q_dim: q_coord}

    #     def _unit_to_sc(u):
    #         if u is None: return None
    #         try: return sc.Unit(str(u))
    #         except Exception: return None

    #     def _bounds_of(p):
    #         low = getattr(p, 'min', getattr(p, 'minimum', None))
    #         high = getattr(p, 'max', getattr(p, 'maximum', None))
    #         b = getattr(p, 'bounds', None)
    #         if (low is None or high is None) and b is not None and len(b) == 2:
    #             low = b[0] if low is None else low
    #             high = b[1] if high is None else high
    #         return low, high

    #     def _fixed_of(p): return bool(getattr(p, 'fixed', False))

    #     store = {}
    #     def new_buf():
    #         return {
    #             'values': sc.zeros(dims=dims_out, shape=shape).values*math.nan,
    #             'vars':   sc.zeros(dims=dims_out, shape=shape).values*math.nan,
    #             'min':    sc.zeros(dims=dims_out, shape=shape).values*math.nan,
    #             'max':    sc.zeros(dims=dims_out, shape=shape).values*math.nan,
    #             'fixed':  sc.zeros(dims=dims_out, shape=shape, dtype='bool').values,
    #             'unit':   None,
                
    #         }

    #     for key, _, _ in specs:
    #         store[key] = new_buf()

    #     # ---- fill all cells ----
    #     for ti in range(t_size):
    #         for qi in range(q_size):
    #             ana = ana_at(ti, qi)
    #             seen = defaultdict(int)
    #             for p in ana.get_parameters():
    #                 occ = seen[p.name]; seen[p.name] += 1

    #                 # register unseen names later in the grid
    #                 if p.name not in name_counts:
    #                     name_counts[p.name] = 1
    #                     if p.name not in store:
    #                         store[p.name] = new_buf()

    #                 k = key_for(p.name, occ) if name_counts[p.name] > 1 else p.name
    #                 if k not in store:  # late duplicate discovery
    #                     k = f'{p.name}[{occ}]'
    #                     if k not in store:
    #                         store[k] = new_buf()

    #                 # set entries
    #                 store[k]['values'][(ti, qi) if temp_dim else (qi,)] = getattr(p, 'value', math.nan)
    #                 err = getattr(p, 'error', None)
    #                 store[k]['vars'][(ti, qi) if temp_dim else (qi,)] = (err**2) if (err is not None) else math.nan
    #                 lo, hi = _bounds_of(p)
    #                 store[k]['min'][(ti, qi) if temp_dim else (qi,)] = lo if lo is not None else math.nan
    #                 store[k]['max'][(ti, qi) if temp_dim else (qi,)] = hi if hi is not None else math.nan
    #                 store[k]['fixed'][(ti, qi) if temp_dim else (qi,)] = _fixed_of(p)

    #                 if store[k]['unit'] is None:
    #                     store[k]['unit'] = _unit_to_sc(getattr(p, 'unit', None))

    #     # ---- build nested DataGroup per parameter ----
    #     out = {}
    #     for key, buf in store.items():
    #         u = buf['unit']

    #         # value (with variances)
    #         val = sc.array(dims=dims_out, values=buf['values'],
    #                     variances=buf['vars'], unit=u) if u is not None \
    #             else sc.array(dims=dims_out, values=buf['values'],
    #                             variances=buf['vars'])
    #         da_value = sc.DataArray(data=val, coords=coords_out)

    #         # min/max share unit; fixed is bool
    #         da_min = sc.DataArray(
    #             sc.array(dims=dims_out, values=buf['min'], unit=u) if u is not None
    #             else sc.array(dims=dims_out, values=buf['min']),
    #             coords=coords_out
    #         )
    #         da_max = sc.DataArray(
    #             sc.array(dims=dims_out, values=buf['max'], unit=u) if u is not None
    #             else sc.array(dims=dims_out, values=buf['max']),
    #             coords=coords_out
    #         )
    #         da_fixed = sc.DataArray(
    #             sc.array(dims=dims_out, values=buf['fixed'], dtype='bool'),
    #             coords=coords_out
    #         )

    #         out[key] = sc.DataGroup({'value': da_value, 'min': da_min, 'max': da_max, 'fixed': da_fixed})

    #     return sc.DataGroup(out)
