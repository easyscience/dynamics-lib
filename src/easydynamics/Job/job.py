from easyscience.job.job import JobBase

from easydynamics.sample import SampleModel
from easydynamics.experiment import Experiment
from easydynamics.analysis import Analysis
from easydynamics.experiment.data import Data

import scipp as sc
import plopp as pp

from collections import defaultdict, Counter

import numpy as np

from itertools import product

import math

class Job(JobBase):
    def __init__(self, name: str, interface=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.name = name
        self._theory = None
        self._resolution_model = None
        self._background_model = None
        self._experiment = None
        self._analysis = []
        self._analysis_meta = None  
        self._summary = None
        self._info = None
        self._fit_parameters = None


    def set_theory(self, theory):
        """ Set the theoretical model.
        """
        if not isinstance(theory, SampleModel):
            raise TypeError("Theory model must be an instance of SampleModel.")
        self._theory = theory

    def set_experiment(self, experiment):
        """ Set the experimental model.
        """
        if not isinstance(experiment, Experiment):
            raise TypeError("Experiment model must be an instance of Experiment.")
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
            raise TypeError("Resolution model must be None or an instance of SampleModel.")
        self._resolution_model = resolution

        if self._resolution_model is not None:
            self.normalize_resolution()

    def normalize_resolution(self):
        """ Normalize the resolution model to have an area of 1.
        """
        self._resolution_model.normalize_area()        

    def append_analysis(self, analysis):
        self._analysis.append(analysis)
        if self._experiment is not None:
            self._analysis[-1].set_experiment(self._experiment)
        if self._theory is not None:
            self._analysis[-1].set_theory(self._theory)

        # Update analysis_meta
        if self._analysis_meta is None:
            self._analysis_meta = {
                'dims': (),
                'sizes': {},
            }

        # Generate metadata dynamically
        current_dims = self._experiment._data.data.dims if self._experiment is not None else []
        energy_dim = 'energy'

        # Exclude the 'energy' dimension and build metadata for other dimensions
        dims_to_track = [d for d in current_dims if d != energy_dim]
        sizes_to_track = {d: self._experiment._data.data.sizes[d] for d in dims_to_track if self._experiment is not None}

        # Update _analysis_meta for dimensionality and sizes
        self._analysis_meta['dims'] = tuple(dims_to_track)
        self._analysis_meta['sizes'] = sizes_to_track            

    def fit(self,
                sequential=None,
                *,
                seed_domains=("theory", "background"),
                copy_offset=False,
                include_temperature=False,
                only_unfixed=True,
                strict_components=True,
                strict_params=True,
                require_same_units=True,
                convert_units=False):
            """

            `_analysis_meta` MUST be set and correct:
                - 'dims': tuple of nesting order (outer -> inner), e.g. ('Temperature','Q') or ('Q',)
                - 'sizes': dict with lengths for each dim
            """

            if not self._analysis:
                raise RuntimeError("No analyses to fit. Build or generate analyses first.")
            
            # If we have just a single analysis object, we don't need to go through a list
            if len(self._analysis) == 1: 
                self._analysis[0].fit()
                return

            if not self._analysis_meta or 'dims' not in self._analysis_meta or 'sizes' not in self._analysis_meta:
                raise RuntimeError("Missing _analysis_meta. Call generate_analysis_for_cuts() or set_analysis_meta().")

            axis_dims: list[str] = list(self._analysis_meta['dims'])
            sizes: dict[str, int] = dict(self._analysis_meta['sizes'])

            # --- helpers ----------------------------------------------------------
            def _walk_all(node):
                """Yield every Analysis in the nested structure."""
                stack = [node]
                while stack:
                    x = stack.pop()
                    if isinstance(x, (list, tuple)):
                        stack.extend(x)
                    else:
                        yield x

            def _get_by_map(idx_map: dict[str, int]):
                """Index self._analysis using axis_dims order and a {dim: idx} map."""
                obj = self._analysis
                for d in axis_dims:
                    if d in idx_map:
                        obj = obj[idx_map[d]]
                return obj

            def _resolve_sweep_dim(tag: str) -> str:
                """Map tags like 'T' to actual dim name found in axis_dims."""
                if tag in axis_dims:
                    return tag
                if tag == 'T':
                    for cand in ('Temperature', 'Temp', 'T'):
                        if cand in axis_dims:
                            return cand
                if tag == 'Q' and 'Q' in axis_dims:
                    return 'Q'
                raise ValueError(f"sweep dim '{tag}' not present in analysis dims {axis_dims}")

            # --- independent fits -------------------------------------------------
            if sequential is None:
                for ana in _walk_all(self._analysis):
                    ana.fit()
                return

            # --- sequential sweeps ------------------------------------------------
            valid = {'Q', '-Q', 'T', '-T'}
            if sequential not in valid:
                raise ValueError(f"sequential must be one of {sorted(valid)} or None, got {sequential!r}")

            backwards = sequential.startswith('-')
            tag = sequential.lstrip('-')
            sweep_dim = _resolve_sweep_dim(tag)

            # outer dims = all other dims in nesting order (could be 0D or >1D)
            outer_dims = [d for d in axis_dims if d != sweep_dim]

            # ranges
            inner_range = range(sizes[sweep_dim]-1, -1, -1) if backwards else range(sizes[sweep_dim])
            outer_ranges = [range(sizes[d]) for d in outer_dims] or [range(1)]

            for outer_combo in product(*outer_ranges):
                idx_map = {}
                for d, i in zip(outer_dims, outer_combo):
                    idx_map[d] = i

                prev_ana = None
                for i in inner_range:
                    idx_map[sweep_dim] = i
                    ana = _get_by_map(idx_map)

                    if prev_ana is not None:
                        ana.seed_from(
                            prev_ana,
                            domains=seed_domains,
                            only_unfixed=only_unfixed,
                            strict_components=strict_components,
                            strict_params=strict_params,
                            include_temperature=include_temperature,
                            require_same_units=require_same_units,
                            convert_units=convert_units,
                            copy_offset=copy_offset,
                        )
                    ana.fit()
                    prev_ana = ana

    def generate_analysis_for_cuts(self, keep=('energy',)):
        """
        Create a nested structure of Analysis objects by cutting the experiment data
        over all dims NOT in `keep` (default keeps 'energy').

        Result:
        self._analysis  -> nested list with one level per cut-dimension, in the
                            same order as they appear in the data (excluding `keep`).
                            e.g. data.dims = ('Temperature','Q','energy')
                                -> self._analysis[ti][qi]
        self._analysis_meta -> metadata about the grid.
        Each Analysis gets:
            - _cut_indices : dict {dim: index}
            - _cut_coords  : dict {dim: coord value/variable}
        """
        data = self._experiment._data.data

        # Normalize `keep` to a tuple
        if isinstance(keep, str):
            keep = (keep,)
        else:
            keep = tuple(keep)

        # Dims we cut over = all dims not kept
        dims_to_cut = [d for d in data.dims if d not in keep]
        sizes = {d: data.sizes[d] for d in dims_to_cut}
        coords = {d: data.coords.get(d, None) for d in dims_to_cut}

        # Helper: make one Analysis for a given index tuple over dims_to_cut
        def make_analysis(idx_tuple):
            # Slice 1D/kept-dims-only spectrum
            da = data
            for d, i in zip(dims_to_cut, idx_tuple):
                da = da[d, i]
            # Build analysis (copy models)
            ana = Analysis(name=f'Analysis{idx_tuple}')
            theory_copy = self._theory.copy()
            if 'Temperature' in da.coords:
                theory_copy.temperature= da.coords['Temperature'].value
                theory_copy._use_detailed_balance = False #TODO users should be allowed to set this

            ana.set_theory(theory_copy)
            if self._background_model is not None:
                ana.set_background_model(self._background_model.copy())
            if self._resolution_model is not None:
                ana.set_resolution_model(self._resolution_model.copy())

            # Attach sliced data via Experiment/Data
            exp = Experiment()
            dat = Data()
            dat.append(da)
            exp.set_data(dat)
            ana.set_experiment(exp)

            # Annotate for traceability
            ana._cut_indices = dict(zip(dims_to_cut, idx_tuple))
            ana._cut_coords = {}
            for d, i in zip(dims_to_cut, idx_tuple):
                c = coords[d]
                if c is None:
                    ana._cut_coords[d] = None
                else:
                    try:
                        # Prefer scalar value if available
                        ana._cut_coords[d] = c[i].value
                    except Exception:
                        # Fall back to Variable/DataArray slice
                        ana._cut_coords[d] = c[i]
            return ana

        # Build a nested list with one level per cut dimension
        def build_level(level, prefix):
            if level == len(dims_to_cut):
                return make_analysis(prefix)
            d = dims_to_cut[level]
            return [build_level(level + 1, prefix + (i,)) for i in range(sizes[d])]

        # Construct the grid (or a single Analysis if there are no cut dims)
        analysis_grid = build_level(0, ())

        # Save results
        self._analysis = analysis_grid
        self._analysis_meta = {
            'dims': tuple(dims_to_cut),       # order of nesting
            'sizes': sizes,                   # size per cut-dim
            'keep': tuple(keep),              # dims left intact in each slice
        }
        return self




    def plot_data_and_model(self,
                            intensity_min=0.0, intensity_max=0.06,
                            energy_min=-0.02, energy_max=0.02,
                            plot_individual_components=True):

        data = self._experiment._data.data
        energy_dim = 'energy'

        # same shape/coords as data
        fit_total = sc.zeros_like(data)
        component_arrays = {} if plot_individual_components else None

        # all non-energy dims (e.g. ['Temperature','Q'] or just ['Q'])
        loop_dims = [d for d in data.dims if d != energy_dim]

        if not loop_dims:
            E = fit_total.coords[energy_dim].values
            ana = self._analysis[0] if isinstance(self._analysis, (list, tuple)) else self._analysis

            if plot_individual_components:
                comps = ana.calculate_individual_components(E)  # dict
                for name, vals in comps.items():
                    if name not in component_arrays:
                        component_arrays[name] = sc.zeros_like(data)
                    component_arrays[name].values = vals
            fit_total.values = ana.calculate_theory(E)

        else:
            ranges = [range(data.sizes[d]) for d in loop_dims]
            for combo in product(*ranges):
                fsel = fit_total
                for d, i in zip(loop_dims, combo):
                    fsel = fsel[d, i]
                E = fsel.coords[energy_dim].values

                ana = self._analysis
                for i in combo:
                    ana = ana[i]

                if plot_individual_components:
                    comps = ana.calculate_individual_components(E)
                    for name, vals in comps.items():
                        if name not in component_arrays:
                            component_arrays[name] = sc.zeros_like(data)
                        csel = component_arrays[name]
                        for d, i in zip(loop_dims, combo):
                            csel = csel[d, i]
                        csel.values = vals
                fsel.values = ana.calculate_theory(E)

        # Build plot group
        data_and_model = {'Data': self._experiment._data.data, 'Model': fit_total}
        if plot_individual_components and component_arrays:
            data_and_model.update(component_arrays)
        data_and_model = sc.DataGroup(data_and_model)

        # Apply energy window
        energy_min = energy_min * sc.Unit('meV')
        energy_max = energy_max * sc.Unit('meV')

        # Styling
        linestyle = {'Data': 'none', 'Model': '-'}
        marker = {'Data': 'o', 'Model': 'none'}
        markerfacecolor = {'Data': 'none', 'Model': 'none'}
        color = {'Data': 'black', 'Model': 'red'}

        if plot_individual_components and component_arrays:
            for name in component_arrays:
                linestyle[name] = '--'
                marker[name] = 'none'
                markerfacecolor[name] = 'none'

        plot = pp.slicer(
            data_and_model['energy', energy_min:energy_max],
            vmin=intensity_min, vmax=intensity_max,
            keep=['energy'],
            linestyle=linestyle,
            marker=marker,
            markerfacecolor=markerfacecolor,
            color=color
        )

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
        self._fit_parameters = self.get_parameters_as_data_group()
        if self._fit_parameters is None:
            raise RuntimeError("No fit parameters found.")

        if parameter_name is not None:
            if parameter_name not in self._fit_parameters:
                raise KeyError(f"Parameter '{parameter_name}' not found in fit parameters. Available parameters: {list(self._fit_parameters.keys())}")
            return pp.slicer(self._fit_parameters[parameter_name]['value'],keep='Q')


    def use_fit_as_resolution(self, job):
        """
        Copy the fitted model from `job` (1D over Q) into every slice of `self`
        that shares the same Q index, across all other dims (e.g. Temperature).
        """
        # --- basic checks
        if not isinstance(job, Job):
            raise TypeError("Job must be an instance of Job.")
        if not getattr(job, '_analysis', None):
            raise RuntimeError("No analysis found in the provided job.")
        if not getattr(self, '_analysis', None):
            raise RuntimeError("No analysis found in 'self'.")
        if not hasattr(self, '_analysis_meta') or not hasattr(job, '_analysis_meta'):
            raise RuntimeError("Both jobs must have _analysis_meta; call generate_analysis_for_cuts() first.")

        # --- meta / dims
        dims_self = tuple(self._analysis_meta.get('dims', ()))
        sizes_self = dict(self._analysis_meta.get('sizes', {}))
        dims_job  = tuple(job._analysis_meta.get('dims', ()))
        sizes_job = dict(job._analysis_meta.get('sizes', {}))

        if 'Q' not in dims_self or 'Q' not in dims_job:
            raise RuntimeError("Both jobs must include 'Q' in their cut dimensions.")

        # Source job must be 1D over Q
        if dims_job != ('Q',):
            raise RuntimeError("Source job is expected to be 1D over Q (job._analysis[q]).")

        nq_self = sizes_self['Q']
        nq_job  = sizes_job['Q']
        if nq_self != nq_job:
            raise RuntimeError("Mismatch in number of Q points between jobs.")

        # Q coord check 
        q_self = self._experiment._data.data.coords.get('Q', None)
        q_job  = job._experiment._data.data.coords.get('Q', None)
        if q_self is None or q_job is None:
            raise RuntimeError("Both jobs must have a 'Q' coordinate in their data.")
        try:
            q_job_cmp = q_job.to(unit=q_self.unit)
        except Exception:
            q_job_cmp = q_job
        if not np.allclose(q_self.values, q_job_cmp.values, rtol=0.0, atol=1e-12):
            raise RuntimeError("Q coordinates differ between jobs.")

        # --- helpers for nested indexing
        def _get_nested(obj, idx_tuple):
            cur = obj
            for i in idx_tuple:
                cur = cur[i]
            return cur

        # Where is Q in the self nesting?
        q_level_self = dims_self.index('Q')
        other_dims = [d for d in dims_self if d != 'Q']
        other_ranges = [range(sizes_self[d]) for d in other_dims]

        # --- main loop: for each Q, copy into all slices over other dims
        for qi in range(nq_self):
            src_ana = job._analysis[qi]  # source is 1D over Q
            # iterate over cartesian product of other dims (or just once if none)
            for combo in (product(*other_ranges) if other_dims else [()]):
                # build full index tuple in the same order as dims_self
                full_idx = [None] * len(dims_self)
                # place Q index
                full_idx[q_level_self] = qi
                # place other dims
                for d, i in zip(other_dims, combo):
                    full_idx[dims_self.index(d)] = i
                tgt_ana = _get_nested(self._analysis, tuple(full_idx))

                tgt_ana.set_resolution_model(src_ana._theory.copy())
                tgt_ana.fix_resolution_parameters()

        return self


    # def generate_diffusion_analysis(self,diffusion_model):

    def generate_empty_analysis_array(self,keep=('energy')):
        """
        Create a nested structure of Analysis objects by cutting the experiment data
        over all dims NOT in `keep` (default keeps 'energy'). No theory or background is added here.

        Result:
        self._analysis  -> nested list with one level per cut-dimension, in the
                            same order as they appear in the data (excluding `keep`).
                            e.g. data.dims = ('Temperature','Q','energy')
                                -> self._analysis[ti][qi]
        self._analysis_meta -> metadata about the grid.
        Each Analysis gets:
            - _cut_indices : dict {dim: index}
            - _cut_coords  : dict {dim: coord value/variable}
        """
        data = self._experiment._data.data

        # Normalize `keep` to a tuple
        if isinstance(keep, str):
            keep = (keep,)
        else:
            keep = tuple(keep)

        # Dims we cut over = all dims not kept
        dims_to_cut = [d for d in data.dims if d not in keep]
        sizes = {d: data.sizes[d] for d in dims_to_cut}
        coords = {d: data.coords.get(d, None) for d in dims_to_cut}

        # Helper: make one Analysis for a given index tuple over dims_to_cut
        def make_analysis(idx_tuple):
            # Slice 1D/kept-dims-only spectrum
            da = data
            for d, i in zip(dims_to_cut, idx_tuple):
                da = da[d, i]
            # Build analysis (copy models)
            ana = Analysis(name=f'Analysis{idx_tuple}')


            # Attach sliced data via Experiment/Data
            exp = Experiment()
            dat = Data()
            dat.append(da)
            exp.set_data(dat)
            ana.set_experiment(exp)

            # Annotate for traceability
            ana._cut_indices = dict(zip(dims_to_cut, idx_tuple))
            ana._cut_coords = {}
            for d, i in zip(dims_to_cut, idx_tuple):
                c = coords[d]
                if c is None:
                    ana._cut_coords[d] = None
                else:
                    try:
                        # Prefer scalar value if available
                        ana._cut_coords[d] = c[i].value
                    except Exception:
                        # Fall back to Variable/DataArray slice
                        ana._cut_coords[d] = c[i]
            return ana

        # Build a nested list with one level per cut dimension
        def build_level(level, prefix):
            if level == len(dims_to_cut):
                return make_analysis(prefix)
            d = dims_to_cut[level]
            return [build_level(level + 1, prefix + (i,)) for i in range(sizes[d])]

        # Construct the grid (or a single Analysis if there are no cut dims)
        analysis_grid = build_level(0, ())

        # Save results
        self._analysis = analysis_grid
        self._analysis_meta = {
            'dims': tuple(dims_to_cut),       # order of nesting
            'sizes': sizes,                   # size per cut-dim
            'keep': tuple(keep),              # dims left intact in each slice
        }
        return self


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
        """
        Collect parameters from every Analysis in self._analysis (nested or flat)
        and return a Scipp DataGroup with per-parameter DataArrays over all
        non-energy dims (e.g. ['Temperature','Q'] or just ['Q']).

        Output shape follows the order of dims in self._experiment._data.data
        excluding 'energy'.
        """
        data = self._experiment._data.data
        energy_dim = 'energy'
        dims_out = [d for d in data.dims if d != energy_dim]
        shape = tuple(data.sizes[d] for d in dims_out)
        coords_out = {d: data.coords[d] for d in dims_out}

        # --- Walk the nested _analysis, yielding (index_tuple, Analysis)
        def _walk(node, idx_prefix=()):
            if isinstance(node, (list, tuple)):
                for i, child in enumerate(node):
                    yield from _walk(child, idx_prefix + (i,))
            else:
                yield idx_prefix, node

        # --- Find the first leaf Analysis to inspect parameter names
        first_leaf = next(_walk(self._analysis))[1]
        first_params = first_leaf.get_parameters()

        # Duplicate-name handling based on the first leaf
        name_counts = Counter(p.name for p in first_params)
        def key_for(name, occ_idx):
            return name if name_counts[name] == 1 else f'{name}[{occ_idx}]'

        # Build a template order (name occurrences) from the first leaf
        template_seen = defaultdict(int)
        specs = []  # (key, base_name, occ_idx)
        for p in first_params:
            occ = template_seen[p.name]
            template_seen[p.name] += 1
            specs.append((key_for(p.name, occ), p.name, occ))

        # Buffers for each parameter key
        def _new_buf():
            return {
                'values': np.full(shape or (1,), np.nan, dtype=float).reshape(shape or (1,)),
                'vars':   np.full(shape or (1,), np.nan, dtype=float).reshape(shape or (1,)),
                'min':    np.full(shape or (1,), np.nan, dtype=float).reshape(shape or (1,)),
                'max':    np.full(shape or (1,), np.nan, dtype=float).reshape(shape or (1,)),
                'fixed':  np.zeros(shape or (1,), dtype=bool).reshape(shape or (1,)),
                'unit':   None,
            }

        store = {key: _new_buf() for key, _, _ in specs}

        # Helpers to read bounds/fixed/unit
        def _bounds_of(p):
            lo = getattr(p, 'min', getattr(p, 'minimum', None))
            hi = getattr(p, 'max', getattr(p, 'maximum', None))
            b = getattr(p, 'bounds', None)
            if (lo is None or hi is None) and b is not None and len(b) == 2:
                lo = b[0] if lo is None else lo
                hi = b[1] if hi is None else hi
            return lo, hi

        def _fixed_of(p):
            return bool(getattr(p, 'fixed', False))

        def _unit_to_sc(u):
            if u is None: return None
            try: return sc.Unit(str(u))
            except Exception: return None

        # --- Fill buffers for every leaf Analysis
        for idx_tuple, ana in _walk(self._analysis):
            # Sanity: idx_tuple must match number of non-energy dims
            if len(dims_out) != len(idx_tuple):
                # If self._analysis nests fewer levels (e.g. only Q), pad to match shape
                # assuming trailing dims vary in the data but not in analyses.
                if len(dims_out) == 1 and len(idx_tuple) == 0:
                    idx_tuple = (0,)
                else:
                    raise RuntimeError(
                        f"Analysis nesting depth {len(idx_tuple)} does not match "
                        f"non-energy dims {dims_out}"
                    )

            seen = defaultdict(int)
            for p in ana.get_parameters():
                occ = seen[p.name]; seen[p.name] += 1

                # Register late-seen names (not present in first leaf)
                if p.name not in name_counts:
                    name_counts[p.name] = 1
                    store.setdefault(p.name, _new_buf())

                key = key_for(p.name, occ) if name_counts[p.name] > 1 else p.name
                if key not in store:  # late duplicate discovery
                    key = f'{p.name}[{occ}]'
                    store.setdefault(key, _new_buf())

                # Fill numeric fields
                store[key]['values'][idx_tuple] = getattr(p, 'value', math.nan)
                err = getattr(p, 'error', None)
                store[key]['vars'][idx_tuple] = (err**2) if (err is not None) else math.nan
                lo, hi = _bounds_of(p)
                store[key]['min'][idx_tuple] = lo if lo is not None else math.nan
                store[key]['max'][idx_tuple] = hi if hi is not None else math.nan
                store[key]['fixed'][idx_tuple] = _fixed_of(p)

                # Unit (first non-None wins)
                if store[key]['unit'] is None:
                    store[key]['unit'] = _unit_to_sc(getattr(p, 'unit', None))

        # --- Build the DataGroup
        out = {}
        for key, buf in store.items():
            u = buf['unit']
            # values with variances
            if u is not None:
                val = sc.array(dims=dims_out or ['_'], values=buf['values'],
                            variances=buf['vars'], unit=u)
            else:
                val = sc.array(dims=dims_out or ['_'], values=buf['values'],
                            variances=buf['vars'])
            da_value = sc.DataArray(val, coords=coords_out if dims_out else {})

            # min/max share unit; fixed is bool
            da_min = sc.DataArray(
                sc.array(dims=dims_out or ['_'], values=buf['min'], unit=u) if u is not None
                else sc.array(dims=dims_out or ['_'], values=buf['min']),
                coords=coords_out if dims_out else {}
            )
            da_max = sc.DataArray(
                sc.array(dims=dims_out or ['_'], values=buf['max'], unit=u) if u is not None
                else sc.array(dims=dims_out or ['_'], values=buf['max']),
                coords=coords_out if dims_out else {}
            )
            da_fixed = sc.DataArray(
                sc.array(dims=dims_out or ['_'], values=buf['fixed'], dtype='bool'),
                coords=coords_out if dims_out else {}
            )

            out[key] = sc.DataGroup({
                'value': da_value,
                'min':   da_min,
                'max':   da_max,
                'fixed': da_fixed,
            })

        return sc.DataGroup(out)