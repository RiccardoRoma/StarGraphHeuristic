from __future__ import annotations
import abc
import numpy as np
from importlib.metadata import version
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Iterable, Sequence
import copy
import os
import pickle
import yaml
import traceback as tb
from networkx import Graph
from qiskit.primitives import BaseEstimator
from qiskit.primitives import Estimator as TerraEstimator
from qiskit.primitives import BackendEstimator as BackendEstimator
from qiskit.providers import BackendV1, BackendV2
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2, FakeManilaV2, FakeFractionalBackend
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, QuantumError
from qiskit_aer.noise.device import basic_device_gate_errors, basic_device_readout_errors

from qiskit.transpiler import PassManager, StagedPassManager, CouplingMap
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_provider.transpiler.passes.scheduling import ALAPScheduleAnalysis, ASAPScheduleAnalysis, PadDynamicalDecoupling
from qiskit.circuit.library import XGate
from qiskit.circuit import QuantumCircuit

import qiskit_ibm_runtime as qir
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime.options.utils import Unset
from dotenv import load_dotenv



class Calibration(metaclass=abc.ABCMeta):
    def __init__(self,
                 name: str,
                 **kwargs) -> None:
        self.name = name

        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self) -> str:
        string_list = []
        for k, v in self.__dict__.items():
            string_list.append(f"{k}={v}")
            
        out = "Calibration(%s)" % ", ".join(string_list)
        return out

    def to_dict(self) -> Dict:
        return copy.deepcopy(self.__dict__)

    def to_pickle(self,
                  fname: str):
        if os.path.isfile(fname):
            raise ValueError("file {} does already exist!".format(fname))

        with open(fname, "wb") as f:
            pickle.dump(self, f)

    @abc.abstractmethod
    def get_filevector(self) -> Tuple[List, List]:
        """
        Define method to write the summarized (shorted) calibration data in a list format

        output: (header, data)
        """

class EstimatorCalibration(Calibration):
    def __init__(self,
                 est_prim_str: str,
                 **kwargs) -> None:
        super().__init__("EstimatorCalibration")
        self._estimator_str = est_prim_str
        kwargs_validated = self._validate_calibration_inputs(est_prim_str, **kwargs)
        for k, v in kwargs_validated.items():
            setattr(self, k, v)

    @property
    def estimator_str(self):
        return self._estimator_str
    
    def _validate_calibration_inputs(self,
                                     est_prim_str: str,
                                     **kwargs) -> dict:
        est_opt = copy.copy(kwargs)
        if est_prim_str == "aer":
            sub_cat = ["transpilation_options", "backend_options", "run_options", "approximation", "skip_transpilation", "abelian_grouping"]
            sub_cat.sort()

            est_opt_keys_sorted = sorted(list(est_opt.keys()))

            if est_opt_keys_sorted != sub_cat:
                raise ValueError("calibration keywords {} do not match required keywords based on estimator string {}!".format(est_opt_keys_sorted, sub_cat))
            transp_opt = est_opt["transpilation_options"]
            if transp_opt is None:
                est_opt["transpilation_options"] = {}
            elif not isinstance(transp_opt, Dict):
                raise ValueError("transpilation options must be a dictionary!")

            backend_opt = est_opt["backend_options"]
            if backend_opt is None:
                est_opt["backend_options"] = {}
            elif not isinstance(backend_opt, Dict):
                raise ValueError("backend options must be a dictionary!")

            run_opt = est_opt["run_options"]
            if run_opt is None:
                est_opt["run_options"] = {}
            elif not isinstance(run_opt, Dict):
                raise ValueError("run options must be a dictionary!")

            circ_opt_lvl = est_opt["transpilation_options"].get("optimization_level", None)
            if circ_opt_lvl is None:
                circ_opt_lvl = 0
                print("No circuit optimization level was chosen. Set it to default {}.".format(circ_opt_lvl))
                est_opt["transpilation_options"]["optimization_level"] = circ_opt_lvl
            if est_opt["skip_transpilation"] and circ_opt_lvl != 0:
                raise ValueError("skip transpilation flag was set to True. Can't do circuit optimization (level = {}) if transpilation is skipped.".format(circ_opt_lvl))

            if "shots" not in est_opt["run_options"].keys():
                if "shots" not in est_opt["backend_options"].keys():
                    shots = None
                else:
                    shots = est_opt["backend_options"].get("shots")
            else:
                shots = est_opt["run_options"].get("shots")

            if shots is None:
                if est_opt["approximation"] is False:
                    print("number of measurement shots is undefined and approximantion flag was set to False!")
                    
                    shots = 1024
                    print("set number of shots to {} instead of using backend default...".format(shots))
            if "shots" in est_opt["run_options"].keys():
                est_opt["run_options"]["shots"] = shots
            elif "shots" in est_opt["backend_options"].keys():
                est_opt["backend_options"]["shots"] = shots
            else:
                print("No shots key found. Add shots key to run options with value {}".format(shots))
                est_opt["run_options"]["shots"] = shots

            abelian_grouping = est_opt["abelian_grouping"]
            if not isinstance(abelian_grouping, bool):
                raise ValueError("Abelian grouping flag must be bool!")


            
        elif est_prim_str == "ibm_runtime":
            sub_cat = ["default_shots", "optimization_level", "resilience_level", "use_simulator", "seed_estimator", "dynamical_decoupling_options", "max_execution_time", "resilience_options", "execution_options", "twirling_options", "environment_options", "simulator_options"]
            sub_cat.sort()

            est_opt_keys_sorted = sorted(list(est_opt.keys()))

            if est_opt_keys_sorted != sub_cat:
                raise ValueError("calibration keywords {} do not match required keywords based on estimator string {}!".format(est_opt_keys_sorted, sub_cat))

            err_mitig_meth = est_opt["resilience_level"]
            if err_mitig_meth is None:
                err_mitig_meth = 0
                print("error mitigation method is undefined. Set it to {} as default.".format(err_mitig_meth))
                est_opt["resilience_level"] = err_mitig_meth

            use_simulator = est_opt["use_simulator"]
            if not isinstance(use_simulator, bool):
                raise TypeError("flag use_simulator must be bool!")
            
            circ_opt_lvl = est_opt["optimization_level"]
            if circ_opt_lvl is None:
                circ_opt_lvl = 0
                print("circuit optimization level is undefined. Set it to {} as default.".format(circ_opt_lvl))
                est_opt["optimization_level"] = circ_opt_lvl
            elif not isinstance(circ_opt_lvl, int):
                raise TypeError("optimization level must be a integer from [0,1] or None!")
            else:
                if circ_opt_lvl < 0:
                    raise ValueError("optimization level must be a integer from [0,1] or None!")
                elif circ_opt_lvl > 1:
                    raise ValueError("optimization level must be a integer from [0,1] or None!")

            if not isinstance(est_opt["dynamical_decoupling_options"], Dict):
                raise TypeError("dynamical decoupling options must be a dictionary")
            if not isinstance(est_opt["resilience_options"], Dict):
                raise TypeError("resilience options must be a dictionary")
            if not isinstance(est_opt["execution_options"], Dict):
                raise TypeError("execution options must be a dictionary")
            if not isinstance(est_opt["twirling_options"], Dict):
                raise TypeError("twirling options must be a dictionary")
            if not isinstance(est_opt["environment_options"], Dict):
                raise TypeError("environment options must be a dictionary")
            if not isinstance(est_opt["simulator_options"], Dict):
                raise TypeError("simulator options must be a dictionary")
            
            if len(est_opt["simulator_options"].keys()) != 1:
                raise ValueError("Detected more than one element in simulator options dictionary! All elements besides key seed_simulator will be ignored. Noise model, coupling map and basis gate set are extracted from backend.")
            if "seed_simulator" not in est_opt["simulator_options"].keys():
                raise ValueError("Missing key seed_simulator in simulator options")
                        
            # if shots is not set, set it to default
            shots = est_opt["default_shots"]
            if shots is None:
                shots = 1024
                print("number of shots is undefined. Set it to {} as default.".format(shots))
                est_opt["default_shots"] = shots
            elif not isinstance(shots, int):
                raise TypeError("Default number of shots must be a positive non-zero integer or None!")
            else:
                if shots <= 0:
                    raise ValueError("Default number of shots must be a positive non-zero integer or None!")

            max_execution_time = est_opt["max_execution_time"]
            if max_execution_time is None:
                max_execution_time = 10800
                print("max. execution time is undefined. Set it to {} as default".format(max_execution_time))
            elif not isinstance(max_execution_time, int):
                raise TypeError("max. execution time must be a positive, non-zero integer in range [1, 10800] or None! The value defines the maximal executation time of the estimator in seconds.")
            else:
                if max_execution_time < 1:
                    raise ValueError("max. execution time must be a positive, non-zero integer in range [1, 10800] or None! The value defines the maximal executation time of the estimator in seconds.")
                elif max_execution_time > 108000:
                    raise ValueError("max. execution time must be a positive, non-zero integer in range [1, 10800] or None! The value defines the maximal executation time of the estimator in seconds.")

        else:
            raise ValueError("estimator string {} does not match any supported string!".format(est_prim_str))
        return est_opt
    
    def to_dict(self) -> dict:
        est_cal_dict = super().to_dict()
        est_prim_str = est_cal_dict.pop("_estimator_str")
        est_cal_dict["estimator_str"] = est_prim_str

        return est_cal_dict
    
    def to_yaml(self,
                fname: str):
        est_cal_dict = self.to_dict()

        if os.path.isfile(fname):
            raise FileExistsError("File {} to save EstimatorCalibration does already exist!".format(fname))
        
        with open(fname, "w") as f:
            yaml.dump(est_cal_dict, f)

    def get_filevector(self) -> Tuple[list, list]:
        """
        Define method to write the summarized (shorted) calibration data in a list format

        output: (header, data)
        """
        header = []
        data = []

        header.append("estimator_str")
        data.append(self.estimator_str)

        err_mitig_meth = None
        circ_opt_lvl = None
        shots = None
        abelian_grouping = False

        if self.estimator_str == "aer":
            err_mitig_meth = 0
            circ_opt_lvl = self.transpilation_options.get("optimization_level")
            shots = self.run_options.get("shots", None)
            if shots is None:
                shots = self.backend_options.get("shots", 0)
            abelian_grouping = self.abelian_grouping
                
        elif self.estimator_str == "ibm_runtime":
            err_mitig_meth = self.resilience_level

            circ_opt_lvl = self.optimization_level
            
            shots = self.default_shots
            # IBM estimator always uses abelian_grouping 
            # https://quantumcomputing.stackexchange.com/questions/34694/is-qiskits-estimator-primitive-running-paulistrings-in-parallel
            abelian_grouping = True

        
        header.append("err_mitigation")
        data.append(err_mitig_meth)

        header.append("circ_optimization")
        data.append(circ_opt_lvl)

        header.append("meas_shots")
        data.append(shots)

        header.append("abelian_grouping")
        data.append(abelian_grouping)

        # check for None types
        for i in range(len(data)):
            val = data[i]
            if val is None:
                data[i] = "None"

        return header, data
    
def _get_estimator_options(est_cal_in: EstimatorCalibration,
                           backend: BackendV2) -> qir.options.EstimatorOptions:
    est_cal = copy.deepcopy(est_cal_in)
    
    if est_cal.estimator_str == "ibm_runtime":
        # Handle qiskit_ibm_runtime.options.utils UnsetType
        if est_cal.seed_estimator is None:
            est_cal.seed_estimator = Unset
        if est_cal.execution_options["rep_delay"] is None:
            est_cal.execution_options["rep_delay"] = Unset
        
        seed_simulator = est_cal.simulator_options["seed_simulator"]
        if seed_simulator is None:
            seed_simulator = Unset
        if est_cal.use_simulator:
            simulator_options = {"seed_simulator": seed_simulator}
            #simulator_options = copy.deepcopy(est_cal.simulator_options)
            noise_model = NoiseModel.from_backend(backend)
            coupling_map = backend.coupling_map
            # coupling_map= [list(t) for t in list(backend.coupling_map.get_edges())
            basis_gates = noise_model.basis_gates
            simulator_options["noise_model"] = noise_model
            simulator_options["coupling_map"] = coupling_map if coupling_map is not None else Unset
            simulator_options["basis_gates"] = basis_gates
            options = qir.options.EstimatorOptions(default_shots = est_cal.default_shots,
                                                   optimization_level=est_cal.optimization_level, 
                                                   resilience_level=est_cal.resilience_level, 
                                                   seed_estimator = est_cal.seed_estimator,
                                                   dynamical_decoupling = est_cal.dynamical_decoupling_options,
                                                   resilience=est_cal.resilience_options,
                                                   execution=est_cal.execution_options,
                                                   max_execution_time=est_cal.max_execution_time, 
                                                   twirling=est_cal.twirling_options,
                                                   simulator=simulator_options,
                                                   environment=est_cal.environment_options,
                                                   experimental = Unset)
        else:
            options = qir.options.EstimatorOptions(default_shots = est_cal.default_shots,
                                                   optimization_level=est_cal.optimization_level, 
                                                   resilience_level=est_cal.resilience_level, 
                                                   seed_estimator = est_cal.seed_estimator,
                                                   dynamical_decoupling = est_cal.dynamical_decoupling_options,
                                                   resilience=est_cal.resilience_options,
                                                   execution=est_cal.execution_options,
                                                   max_execution_time=est_cal.max_execution_time, 
                                                   twirling=est_cal.twirling_options,
                                                   environment=est_cal.environment_options,
                                                   experimental = Unset)
        return options
        
    else:
        raise ValueError("Estimator calibration string does not match expected string!")
    

def get_estimator(est_cal: EstimatorCalibration,
                  mode: Union[qir.Session, qir.Batch, BackendV2, None] = None) -> BaseEstimator:
    if est_cal.estimator_str == "aer":
        est = AerEstimator(backend_options=est_cal.backend_options, 
                           transpile_options=est_cal.transpilation_options, 
                           run_options=est_cal.run_options, 
                           approximation=est_cal.approximation, 
                           skip_transpilation=est_cal.skip_transpilation, 
                           abelian_grouping=est_cal.abelian_grouping)
        return est
    elif est_cal.estimator_str == "ibm_runtime":
        # extract backend
        if isinstance(mode, qir.Session):
            backend = mode._backend
        elif isinstance(mode, qir.Batch):
            backend = mode.backend()
        elif isinstance(mode, BackendV2):
            backend = mode
        else:
            raise TypeError("Unknown type for mode variable!")
        options = _get_estimator_options(est_cal, backend)
        est = qir.EstimatorV2(mode=mode, options=options)
        return est
        
    else:
        raise ValueError("estimator string {} does not match any supported string!".format(est_cal.estimator_str))
    

def get_EstimatorCalibration_from_dict(est_cal_dict_in: dict) -> EstimatorCalibration:
    est_cal_dict = copy.deepcopy(est_cal_dict_in)
    est_prim_str = est_cal_dict.pop("estimator_str", None)
    if est_prim_str is None:
        raise ValueError("could not retrieve estimator string from file!")
    elif est_prim_str == "ibm_runtime":
        dd_options = est_cal_dict.get("dynamical_decoupling_options", None)
        if dd_options is None:
            est_cal_dict["dynamical_decoupling_options"] = {"enable": False, 
                                                            "sequence_type": "XX",
                                                            "extra_slack_distribution": "middle",
                                                            "scheduling_method": "alap"}
    
    name = est_cal_dict.pop("name", None)

    est_cal = EstimatorCalibration(est_prim_str, **est_cal_dict)
    return est_cal

def get_EstimatorCalibration_from_yaml(fname: str) -> EstimatorCalibration:
    if not os.path.isfile(fname):
        raise ValueError("file {} does not exist!".format(fname))

    cal_dict = None
    raw_data = None
    with open(fname, "r") as f:
        raw_data = f.read()

    cal_dict = yaml.load(raw_data, Loader=yaml.Loader)
    if cal_dict is None:
        raise ValueError("Something went wrong while reading in yaml text file! resulting dictionary is empty!")
    
    est_cal_dict = cal_dict.get("estimator_calibration", None)
    if est_cal_dict is None:
        raise ValueError("Something went wrong while reading in yaml text file! No estimator calibration subdictionary found!")
    
    use_simulator = est_cal_dict.get("use_simulator", None)
    if use_simulator is None:
        est_cal_dict["use_simulator"] = cal_dict["run_locally"]

    return get_EstimatorCalibration_from_dict(est_cal_dict)
    

def get_EstimatorCalibration_from_pickle(fname: str) -> EstimatorCalibration:
    if not os.path.isfile(fname):
        raise ValueError("file {} does not exist!".format(fname))

    est_cal = None
    with open(fname, "rb") as f:
        est_cal = pickle.load(f)

    if not isinstance(est_cal, EstimatorCalibration):
        raise TypeError("loaded pickle object is no EstimatorCalibration!")

    return est_cal


class PresetPassManagerCalibration(Calibration):
    def __init__(self,
                 optimization_level: int = 0,
                 initial_layout: Union[List[int], None]=None,
                 layout_method: Union[str, None] = None,
                 routing_method: Union[str, None] = None,
                 translation_method: Union[str, None] = None,
                 scheduling_method: Union[str, None] = None,
                 approximation_degree: float = 1.0,
                 seed_transpile: Union[int, None] = None,
                 unitary_synthesis_method: str = 'default',
                 unitary_synthesis_plugin_config: Union[dict, None] = None,
                 dd_options: Union[dict, None] = None,
                 transpilation_trials: int = 1):
        super().__init__("PresetPassManagerCalibration")
        
        self._validate_inputs(optimization_level, layout_method, routing_method, translation_method, scheduling_method, approximation_degree, dd_options, transpilation_trials)
        self.optimization_level = optimization_level
        self.initial_layout = initial_layout
        self.layout_method = layout_method
        self.routing_method = routing_method
        self.translation_method = translation_method
        self.scheduling_method = scheduling_method
        self.approximation_degree = approximation_degree
        self.seed_transpile = seed_transpile
        self.unitary_synthesis_method = unitary_synthesis_method
        self.unitary_synthesis_plugin_config = unitary_synthesis_plugin_config
        self.dynamical_decoupling_options = dd_options
        self.transpilation_trials = transpilation_trials
    
    def __repr__(self) -> str:

        string_list = []
        for k, v in self.__dict__.items():
            if k != "name":
                string_list.append(f"{k}={v}")

        out = "PresetPassManagerCalibration(%s)" % ", ".join(string_list)
        return out

    def to_yaml(self,
                fname: str):
        # convert to dictionary
        cal_dict = self.to_dict()
        
        # check if file already exists
        if os.path.isfile(fname):
            raise ValueError("file {} does already exist!".format(fname))

        # dump calibration dictionary into yaml file
        with open(fname, "w") as f:
            yaml.dump(cal_dict, f)
                        
    def get_filevector(self) -> Tuple[List, List]:
        """
        Define method to write the summarized (shorted) calibration data in a list format

        output: (header, data)
        """

        header = []
        data = []

        for k, v in self.__dict__.items():
            if k == "name":
                pass
            elif k == "dynamical_decoupling_options":
                # unpack dd_options
                if v is not None:
                    for key, val in v.items():
                        header.append("dd_"+key)
                        if val is not None:
                            data.append(val)
                        else:
                            data.append("None")
            else:
                header.append(k)
                if v is not None:
                    data.append(v)
                else:
                    data.append("None")
        return (header, data)
    
    def _validate_inputs(self,
                         optimization_level: int = 0,
                         layout_method: Union[str, None] = None,
                         routing_method: Union[str, None] = None,
                         translation_method: Union[str, None] = None,
                         scheduling_method: Union[str, None] = None,
                         approximation_degree: float = 1.0,
                         dd_options: Union[dict, None] = None,
                         transpilation_trials: int = 1):
        if not isinstance(optimization_level, int):
            raise ValueError("Optimization level must be one of the integers 0, 1, 2, 3!")
        
        if (optimization_level < 0) or (optimization_level > 3):
            raise ValueError("optimization level must be either 0, 1, 2 or 3!")
        
        valid_layout_methods = ['trivial', 'dense', 'sabre']
        if layout_method is not None:
            if layout_method not in valid_layout_methods:
                raise ValueError("Layout method {} is invalid! Method must be one of the valid strings: {}!".format(layout_method, valid_layout_methods))

        valid_routing_methods = ['basic', 'lookahead', 'stochastic', 'sabre', 'none']
        if routing_method is not None:
            if routing_method not in valid_routing_methods:
                raise ValueError("Routing method {} is invalid! Method must be one of the valid strings: {}!".format(routing_method, valid_routing_methods))

        valid_translation_methods = ['translator', 'synthesis']
        if translation_method is not None:
            if translation_method not in valid_translation_methods:
                 raise ValueError("Translation method {} is invalid! Method must be one of the valid strings: {}!".format(translation_method, valid_translation_methods))

        
        valid_scheduling_methods = ['alap', 'asap']
        if scheduling_method is not None:
            if scheduling_method not in valid_scheduling_methods:
                 raise ValueError("Scheduling method {} is invalid! Method must be one of the valid strings: {}!".format(scheduling_method, valid_scheduling_methods))
                
        if (approximation_degree > 1.0) or (approximation_degree < 0.0):
            raise ValueError("Approximation degree {} is invalid! Must be within the interval [0.0, 1.0].".format(approximation_degree))
        if not isinstance(approximation_degree, float):
            raise ValueError("Approximation degree must be a float number within [0.0, 1.0]!")
        
        if dd_options is not None:
            if isinstance(dd_options, dict):
                enable = dd_options["enable"]
                if not isinstance(enable, bool):
                    raise ValueError("flag enable for dynamical decoupling options must be bool!")
                dd_schedule = dd_options["scheduling_method"]
                if dd_schedule not in valid_scheduling_methods:
                    raise ValueError("Scheduling method {} in dynamical decoupling options is invalid! Method must be one of the valid strings: {}!".format(dd_schedule, valid_scheduling_methods))
                if enable:
                    if scheduling_method is not None:
                        if scheduling_method != dd_schedule:
                            print("Warning: dynamical decouling is activated and scheduling method for preset passmanager function differs from DD scheduling method! DD will overwrite scheduling method from preset passmanger.")
                dd_sequence = dd_options["sequence_type"]
                valid_sequence_types = ["XX"]
                if dd_sequence not in valid_sequence_types:
                    raise ValueError("Gate sequence {} for dynamical decoupling is invalid! Method must be one of the valid strings: {}!".format(dd_sequence, valid_sequence_types))
        
        if not isinstance(transpilation_trials, int):
            raise ValueError("Number of transpilation trials must be a positive, non-zero integer!")
        elif transpilation_trials <= 0:
            raise ValueError("Number of transpilation trials must be a positive, non-zero integer!")
            
def get_PresetPassManagerCalibration_from_dict(cal_dict: dict) -> PresetPassManagerCalibration:

    opt_lvl = cal_dict.get("optimization_level", None)
    if opt_lvl is None:
        print("Could not retrieve optimization level! Set it to default value 0")
        opt_lvl = 0

    initial_layout = cal_dict.get("initial_layout", None)
    layout_method = cal_dict.get("layout_method", None)
    routing_method = cal_dict.get("routing_method", None)
    translation_method = cal_dict.get("translation_method", None)
    scheduling_method = cal_dict.get("scheduling_method", None)
    approximation_degree = cal_dict.get("approximation_degree", None)
    if approximation_degree is None:
        print("Could not retrieve approximation degree! Set it to default value 1.0")
        approximation_degree=1.0

    seed_transpile = cal_dict.get("seed_transpile", None)

    unitary_synthesis_method = cal_dict.get("unitary_synthesis_method", None)
    if unitary_synthesis_method is None:
        print("Could not retrieve unitary synthesis method! Set it to default")
        unitary_synthesis_method = "default"

    unitary_synthesis_plugin_config = cal_dict.get("unitary_synthesis_plugin_config", None)

    dd_options = cal_dict.get("dynamical_decoupling_options", None)

    transpilation_trials = cal_dict.get("transpilation_trials", 1)

    return PresetPassManagerCalibration(optimization_level=opt_lvl, 
                                        initial_layout=initial_layout, 
                                        layout_method=layout_method, 
                                        routing_method=routing_method, 
                                        translation_method=translation_method,
                                        scheduling_method=scheduling_method, 
                                        approximation_degree=approximation_degree,
                                        seed_transpile=seed_transpile,
                                        unitary_synthesis_method=unitary_synthesis_method,
                                        unitary_synthesis_plugin_config=unitary_synthesis_plugin_config,
                                        dd_options=dd_options,
                                        transpilation_trials = transpilation_trials)

def get_PresetPassManagerCalibration_from_yaml(fname: str) -> PresetPassManagerCalibration:
    
    if not os.path.isfile(fname):
        raise ValueError("file {} does not exist!".format(fname))

    cal_dict = None
    raw_data = None
    with open(fname, "r") as f:
        raw_data = f.read()

    cal_dict = yaml.load(raw_data, Loader=yaml.Loader)
    if cal_dict is None:
        raise ValueError("Something went wrong while reading in yml text file! resulting dictionary is empty!")
    
    pm_cal_dict = cal_dict.get("passmanager_calibration", None)

    if pm_cal_dict is None:
        raise ValueError("Something went wrong while reading in yml text file! No passmanager subdictionary found!")
    
    
    
    return get_PresetPassManagerCalibration_from_dict(pm_cal_dict)
    

def get_PresetPassManagerCalibration_from_pickle(fname: str) -> PresetPassManagerCalibration:
    if not os.path.isfile(fname):
        raise ValueError("file {} does not exist!".format(fname))

    pm_cal = None
    with open(fname, "rb") as f:
        pm_cal = pickle.load(f)

    if not isinstance(pm_cal, PresetPassManagerCalibration):
        raise ValueError("loaded pickle object is no PresetPassManagerCalibration!")

    return pm_cal

def get_passmanager(backend: BackendV2,
                    pm_cal: PresetPassManagerCalibration) -> PassManager:

    # construct preset passmanager
    staged_passmanager = generate_preset_pass_manager(optimization_level=pm_cal.optimization_level, 
                                        backend=backend,
                                        initial_layout=pm_cal.initial_layout,
                                        layout_method=pm_cal.layout_method,
                                        routing_method=pm_cal.routing_method,
                                        translation_method=pm_cal.translation_method,
                                        scheduling_method=pm_cal.scheduling_method,
                                        approximation_degree=pm_cal.approximation_degree,
                                        seed_transpiler=pm_cal.seed_transpile,
                                        unitary_synthesis_method=pm_cal.unitary_synthesis_method,
                                        unitary_synthesis_plugin_config=pm_cal.unitary_synthesis_plugin_config)
    # get dynamical decoupling options
    dynamical_decoupling_opt = pm_cal.dynamical_decoupling_options

    if dynamical_decoupling_opt["enable"]:
        #durations = InstructionDurations().from_backend(backend)
        durations = backend.instruction_durations
        if dynamical_decoupling_opt["sequence_type"] == "XX":
            dd_sequence = [XGate(), XGate()]
        else:
            raise NotImplementedError("Dynamical decoupling sequence string {} is not supported.".format(dynamical_decoupling_opt["sequence_type"]))
        
        if dynamical_decoupling_opt["scheduling_method"] == "alap":
            scheduling_pass = ALAPScheduleAnalysis(durations)
        elif dynamical_decoupling_opt["scheduling_method"] == "asap":
            scheduling_pass = ASAPScheduleAnalysis(durations)
        else:
            raise NotImplementedError("Scheduling method {} is not supported.".format(dynamical_decoupling_opt["scheduling_method"]))
        
        passmanager_dd = PassManager([scheduling_pass, PadDynamicalDecoupling(durations, dd_sequence)])
    
        # append dd passmanager to preset passmanager (more robust than applying it afterwards seperately)
        staged_passmanager.scheduling = passmanager_dd
    
    return staged_passmanager

# load IBM Quantum credentials
def load_ibm_credentials(premium_access: bool = False) -> QiskitRuntimeService:
    load_dotenv()
    ibmq_api_token = os.environ["TOKEN"]
    if premium_access:
        ibmq_hub = os.environ["HUB"]
        ibmq_group = os.environ["GROUP"]
        ibmq_project = os.environ["PROJECT"]
    else:
        ibmq_hub = os.environ["HUB_open"]
        ibmq_group = os.environ["GROUP_open"]
        ibmq_project = os.environ["PROJECT_open"]
    
    service = QiskitRuntimeService(channel="ibm_quantum", token=ibmq_api_token, instance=f"{ibmq_hub}/{ibmq_group}/{ibmq_project}")
    return service

def get_simple_noise_model_from_backend(service: QiskitRuntimeService,
                                        backend_str: str, 
                                        consider_2qubit_gate_error: bool = True,
                                        consider_readout_error: bool = True) -> NoiseModel:
    """
    This function simplifies the backend noise model which is derived from given backend_str.
    The simplified noise model considers only the two-qubit gate error and/or the qubit readout error.
    """
    backend_opt = {"backend_str": backend_str,
                   "noise_model_id": 0,
                   "fname_noise_model": "",
                   "noise_model_str": "",
                   "coupling_map_id": 0,
                   "fname_coupling_map": "",
                   "coupling_map_str": "",
                   "native_basis_gates_str": "",
                   "run_locally": False}
    
    device_backend = get_backend(service, backend_opt, print_status=False)
    noise_model = NoiseModel(basis_gates=device_backend.operation_names)

    if consider_2qubit_gate_error:
        gate_errs = basic_device_gate_errors(target=device_backend.target)
    
        # check available two-qubit gate
        if "cx" in device_backend.operation_names:
            two_qubit_gate_str = "cx"
        elif "ecr" in device_backend.operation_names:
            two_qubit_gate_str = "ecr"
        else:
            raise ValueError("Cannot find two qubit gate in backend operations!")
    
        two_qubit_gate_errs = []
        for et in gate_errs:
            if et[0] == two_qubit_gate_str:
                two_qubit_gate_errs.append(et)
        
        for et in two_qubit_gate_errs:
            noise_model.add_quantum_error(et[2], et[0], et[1])

    if consider_readout_error:
        read_errs = basic_device_readout_errors(target=device_backend.target)

        for et in read_errs:
            noise_model.add_readout_error(et[1], et[0])

    return noise_model
    


def get_backend(service: QiskitRuntimeService,
                backend_opt: dict,
                print_status: bool = True) -> BackendV2:
    backend_str = backend_opt['backend_str']

    noise_model_id = backend_opt['noise_model_id']
    fname_noise_model = backend_opt['fname_noise_model']
    noise_model_str = backend_opt['noise_model_str']

    coupling_map_id = backend_opt['coupling_map_id']
    fname_coupling_map = backend_opt['fname_coupling_map']
    coupling_map_str = backend_opt['coupling_map_str']

    native_basis_gates_str = backend_opt['native_basis_gates_str']
    # setup backend
    if backend_str == "aer_simulator":
        if print_status:
            print("Start simulation runs with local Aer simulator...")
        # load possible noise_model, coupling map and basis gate set
        if noise_model_id == -1:
    
            if fname_noise_model == "":
                raise ValueError("noise model id -1 means to read model from pickle file but file name is empty!")
            if os.path.isfile(fname_noise_model):
                with open(fname_noise_model, "rb") as f:
                    noise_model = pickle.load(f)
                if print_status:
                    print("Loaded noise model from file!")
            else:
                raise ValueError("pickle file to read noise model does not exist!")
            if noise_model_str == "":
                raise ValueError("Forgot to define a noise model string for file model!")
        elif noise_model_id==0:
            noise_model_str = "None"
            noise_model = None
            if print_status:
                print("No noise model is used!")
        elif noise_model_id==1:
            if "fake" in noise_model_str:
                device_backend = FakeProviderForBackendV2().backend(noise_model_str)
            else:
                device_backend = service.backend(noise_model_str)
            noise_model = NoiseModel.from_backend(device_backend)
            if print_status:
                print("Loaded noise model from backend {}!".format(noise_model_str))
        elif noise_model_id==2:
            # load simplified noise model from backend noise_model_str, considering only two-qubit gate errors
            noise_model = get_simple_noise_model_from_backend(service, noise_model_str, consider_2qubit_gate_error=True, consider_readout_error=False)
            if print_status:
                print("Loaded simplified noise model from backend {}, considering only two-qubit gate errors!".format(noise_model_str))
        elif noise_model_id==3:
            # load simplified noise model from backend noise_model_str, considering only readout errors
            noise_model = get_simple_noise_model_from_backend(service, noise_model_str, consider_2qubit_gate_error=False, consider_readout_error=True)
            if print_status:
                print("Loaded simplified noise model from backend {}, considering only readout errors!".format(noise_model_str))
        elif noise_model_id==4:
            # load simplified noise model from backend noise_model_str, considering only two-qubit gate errors and readout errors
            noise_model = get_simple_noise_model_from_backend(service, noise_model_str, consider_2qubit_gate_error=True, consider_readout_error=True)
            if print_status:
                print("Loaded simplified noise model from backend {}, considering only two-qubit gate errors and readout errors!".format(noise_model_str))
        else:
            raise ValueError("noise model id {} is currently not supported.".format(noise_model_id))
        
        if coupling_map_id == -1:
            if fname_coupling_map == "":
                raise ValueError("coupling map id -1 means to read coupling map from pickle file but file name is empty!")
            if os.path.isfile(fname_coupling_map):
                with open(fname_coupling_map, "rb") as f:
                    cm_object = pickle.load(f)
                if isinstance(cm_object, Graph):
                    # convert networkx graph to coupling map
                    coupling_map = CouplingMap(list(cm_object.edges))
                elif isinstance(cm_object, CouplingMap):
                    coupling_map = cm_object
                elif isinstance(cm_object, list):
                    # convert list of edges to coupling map
                    coupling_map = CouplingMap(cm_object)
                else:
                    raise ValueError("Unkown type for coupling map from pickle file.")
                if print_status:
                    print("Loaded coupling map from file!")
            else:
                raise ValueError("pickle file to read coupling map does not exist!")
        
            if coupling_map_str == "":
                raise ValueError("Forgot to define a coupling map string for map from file!")
        elif coupling_map_id==0:
            coupling_map_str = "None"
            coupling_map = None
            if print_status:
                print("No coupling map is used!")
        elif coupling_map_id==1:
            if "fake" in coupling_map_str:
                device_backend = FakeProviderForBackendV2().backend(coupling_map_str)
            else:
                device_backend = service.backend(coupling_map_str)

            if device_backend.coupling_map:
                # load coupling map from backend if not None
                coupling_map = device_backend.coupling_map
            else:
                # load coupling map from target
                if device_backend.target.build_coupling_map():
                    coupling_map = device_backend.target.build_coupling_map()
                else:
                    # check available two-qubit gate
                    if "cx" in device_backend.operation_names:
                        two_qubit_gate_str = "cx"
                    elif "ecr" in device_backend.operation_names:
                        two_qubit_gate_str = "ecr"
                    else:
                        raise ValueError("Cannot find two qubit gate in backend operations!")
                    if device_backend.target.build_coupling_map(two_q_gate=two_qubit_gate_str):
                        coupling_map = device_backend.target.build_coupling_map(two_q_gate=two_qubit_gate_str)
                    else:
                        raise ValueError("Unable to load a valid coupling map from backend {}.".format(coupling_map_str))
                    
            if print_status:
                print("Loaded coupling map from backend {}!".format(coupling_map_str))
        else:
            raise ValueError("coupling map id {} is currently not supported.".format(coupling_map_id))
        
        if native_basis_gates_str:
            if "fake" in native_basis_gates_str:
                device_backend = FakeProviderForBackendV2().backend(native_basis_gates_str)
            else:
                device_backend = service.backend(native_basis_gates_str)
            native_basis_gates = NoiseModel.from_backend(device_backend).basis_gates
            if print_status:
                print("Loaded basis gate set from backend {}!".format(native_basis_gates_str))
        else:
            if noise_model is None:
                native_basis_gates = None
                if print_status:
                    print("No basis gate set is used!")
            else:
                native_basis_gates = noise_model.basis_gates
                if print_status:
                    print("Loaded basis gate set from noise model!")

        # limit number of qubits to coupling map
        if coupling_map is not None:
            num_qubits = len(coupling_map.physical_qubits)
            backend = AerSimulator(method="automatic", n_qubits=num_qubits, noise_model=noise_model, coupling_map=coupling_map, basis_gates=native_basis_gates)
        else:
            backend = AerSimulator(method="automatic", noise_model=noise_model, coupling_map=coupling_map, basis_gates=native_basis_gates)
    elif "fake" in backend_str:
        # consistency check
        if noise_model_id != 0:
            print("backend string does not equal aer_simulator but noise model is not None. Use given backend noise model!")
        if coupling_map_id != 0:
            print("backend string does not equal aer_simulator but coupling map is not None. Use given backend coupling map!")
        if native_basis_gates_str:
            print("backend string does not equal aer_simulator but native basis gates string is not empty. Use given backend native basis gate set!")
    
        try:
            backend = FakeProviderForBackendV2().backend(name=backend_str)
            if print_status:
                print("Start simulation runs with fake ibm backend simulator...")
                print("Load noise model, coupling map and basis gate set from fake backend {}!".format(backend_str))
        except Exception as exc:
                print(f"Loading of fake ibm runtime backend {backend_str} failed! Backend is not found with FakeProviderForBackendV2.")
                print("Exception message:")
                tb.print_exc()
    else:
        # consistency check
        if noise_model_id != 0:
            print("backend string does not equal aer_simulator but noise model is not None. Use given backend noise model!")
        if coupling_map_id != 0:
            print("backend string does not equal aer_simulator but coupling map is not None. Use given backend coupling map!")
        if native_basis_gates_str:
            print("backend string does not equal aer_simulator but native basis gates string is not empty. Use given backend native basis gate set!")
    
        if backend_opt["run_locally"]:
            # simulate hardware backend with aer simulator locally
            backend = AerSimulator.from_backend(service.backend(backend_str))
            if print_status:
                print("Start simulation runs with real ibm backend in local testing mode with Aer simulator...")
                print("Load noise model, coupling map and basis gate set from real backend {}!".format(backend_str))
        else:
            # run on actual hardware
            backend = service.backend(backend_str)
            if print_status:
                print("Start simulation runs on real ibm backend {} (hardware run)...".format(backend_str))
    return backend
    
# (transpile circuits?)
def transpile_circuits(staged_passmanager: StagedPassManager,
                       circuits: List[QuantumCircuit],
                       transpilation_trials: int = 1,
                       remove_barriers: bool = True) -> List[QuantumCircuit]:
    
    # remove all barriers before transpilation if desired
    if remove_barriers:
        circs_transp = [RemoveBarriers()(c) for c in circuits]
    else:
        circs_transp = [c.copy() for c in circuits]
    
    # transpile circuits to backend via staged passmanager
    curr_transp_circs = []
    for c in circs_transp:
        curr_best = staged_passmanager.run(c)
        # consider also transpilation trials to get the smallest circuit depth out of all trials
        for _ in range(transpilation_trials-1):
            curr_transp = staged_passmanager.run(c)
            if curr_transp.depth() < curr_best.depth():
                curr_best = curr_transp.copy()
                
        # check if if_else blocks have been transpiled correctly, and fix it if not
        verify_if_else_params_in_transp_circ(curr_best)

        ## To-Do: add options to check that final layout (circuit.layout) corresponds to initial layout

        curr_transp_circs.append(curr_best)
    circs_transp = [c.copy() for c in curr_transp_circs]

    return circs_transp

    
def verify_if_else_params_in_transp_circ(circ: QuantumCircuit):
    virtual_to_physical_map = circ.layout.initial_layout.get_virtual_bits()
    for d in circ.data:
        if d.operation.name == "if_else":
            for i in range(len(d.operation.params)):
                p = d.operation.params[i]
                if p is not None:
                    new_instrs = []
                    for cinstr in p.data:
                        if not set(cinstr.qubits).issubset(circ.qubits):
                            curr_qubits = cinstr.qubits
                            new_qubits = tuple([circ.qubits[virtual_to_physical_map[q]] for q in curr_qubits])
                            new_instrs.append(cinstr.replace(qubits=new_qubits))
                        else:
                            new_instrs.append(cinstr)
                    d.operation.params[i] = QuantumCircuit.from_instructions(new_instrs)
                        
