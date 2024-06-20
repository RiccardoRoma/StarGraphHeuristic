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
from qiskit.primitives import BaseEstimator
from qiskit.primitives import Estimator as TerraEstimator
from qiskit.primitives import BackendEstimator as BackendEstimator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.noise import NoiseModel

from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
 

import qiskit_ibm_runtime as qir
from qiskit_ibm_provider import IBMProvider
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
                 est_opt: Dict,
                 noise_model_str: str,
                 coupling_map_str: str,
                 basis_gates_str: str,
                 est_prim_str: str,
                 backend_str: str) -> None:
        super().__init__("EstimatorCalibration")
        self._estimator_options = self._validate_estimator_options(est_opt, est_prim_str)
        self.noise_model_str = noise_model_str
        self.coupling_map_str = coupling_map_str
        self.basis_gates_str = basis_gates_str
        self._estimator_str = est_prim_str
        self.backend_str = backend_str

    @property
    def estimator_options(self):
        return self._estimator_options
    @estimator_options.setter
    def estimator_options(self,
                          est_opt: Dict):
        self._estimator_options = self._validate_estimator_options(est_opt, self.estimator_str)

    @property
    def estimator_str(self):
        return self._estimator_str

    def __repr__(self) -> str:
        out = "EstimatorCalibration(est_opt={}, noise_model_str={}, coupling_map_str={}, basis_gates_str={}, est_prim_str={}, backend_str={})".format(self._estimator_options, self.noise_model_str, self.coupling_map_str, self.basis_gates_str, self.estimator_str, self.backend_str)

        return out

    def to_dict(self):
        est_cal_dict = super().to_dict()
        est_str = est_cal_dict.pop("_estimator_str")
        est_cal_dict["estimator_str"] = est_str

        est_opt = est_cal_dict.pop("_estimator_options")
        est_cal_dict["estimator_options"] = est_opt

        return est_cal_dict

    def to_yaml(self,
                fname: str):
        # convert to dictionary
        est_cal_dict = self.to_dict()
        # search for noise_model (should not be contained in the yaml but pickled 
        for key in est_cal_dict["estimator_options"].keys():
            # check only the dictionaries in estimator options
            if isinstance(est_cal_dict["estimator_options"][key], Dict):
                # check if noise_model key exists
                noise_model = est_cal_dict["estimator_options"][key].pop("noise_model", None)
                # if noise_model is not None, replace with noise_model_str in yaml (when loaded this will be again replaced by pickled noise_model)
                if noise_model is not None:

                    fname_noise_model, yaml_ext = os.path.splitext(fname)
                    fname_noise_model = fname_noise_model + "_noise_model.pickle"
                    
                    if os.path.isfile(fname_noise_model):
                        raise ValueError("file for saving noise_model {} does already exist!".format(fname_noise_model))
                    with open(fname_noise_model, "wb") as f:
                        pickle.dump(noise_model, f)

                    # est_cal_dict["estimator_options"][key]["noise_model"] = est_cal_dict["noise_model_str"]
                    est_cal_dict["estimator_options"][key]["noise_model"] = fname_noise_model
                    
        # check if file already exists
        if os.path.isfile(fname):
            raise ValueError("file {} does already exist!".format(fname))

        # dump calibration dictionary into yaml file
        with open(fname, "w") as f:
            yaml.dump(est_cal_dict, f)
            
    def _validate_estimator_options(self,
                                    est_opt_in: Dict,
                                    est_prim_str: str) -> Dict:
        est_opt = copy.copy(est_opt_in)
        if est_prim_str == "aer":
            sub_cat = ["transpilation_options", "backend_options", "run_options", "approximation", "skip_transpilation", "abelian_grouping"]
            sub_cat.sort()

            est_opt_keys_sorted = sorted(list(est_opt.keys()))

            if est_opt_keys_sorted != sub_cat:
                raise ValueError("estimator options dictionaries sub-catagories {} do not match required sub-catagories based on estimator string {}!".format(est_opt_keys_sorted, sub_cat))
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
            sub_cat = ["optimization_level", "resilience_level", "max_execution_time", "transpilation_options", "resilience_options", "execution_options", "environment_options", "simulator_options"]
            sub_cat.sort()

            est_opt_keys_sorted = sorted(list(est_opt.keys()))

            if est_opt_keys_sorted != sub_cat:
                raise ValueError("estimator options dictionaries sub-catagories {} do not match required sub-catagories based on estimator string {}!".format(est_opt_keys_sorted, sub_cat))

            err_mitig_meth = est_opt["resilience_level"]
            if err_mitig_meth is None:
                err_mitig_meth = 0
                print("error mitigation method is undefined. Set it to {} as default.".format(err_mitig_meth))
                est_opt["resilience_level"] = err_mitig_meth
            
            circ_opt_lvl = est_opt["optimization_level"]
            if circ_opt_lvl is None:
                circ_opt_lvl = 0
                print("circuit optimization level is undefined. Set it to {} as default.".format(circ_opt_lvl))
                est_opt["optimization_level"] = circ_opt_lvl

            if not isinstance(est_opt["transpilation_options"], Dict):
                raise ValueError("transpilation options must be a dictionary")
            if not isinstance(est_opt["resilience_options"], Dict):
                raise ValueError("resilience options must be a dictionary")
            if not isinstance(est_opt["execution_options"], Dict):
                raise ValueError("execution options must be a dictionary")
            if not isinstance(est_opt["environment_options"], Dict):
                raise ValueError("environment options must be a dictionary")
            if not isinstance(est_opt["simulator_options"], Dict):
                raise ValueError("simulator options must be a dictionary")
            
            
            
            
            # if shots is not set, set it to default
            shots = est_opt["execution_options"].get("shots", None)
            if shots is None:
                shots = 1024
                print("number of shots is undefined. Set it to {} as default.".format(shots))
                est_opt["execution_options"]["shots"] = shots


        elif est_prim_str == "terra":
            sub_cat = ["run_options"]
            sub_cat.sort()

            est_opt_keys_sorted = sorted(list(est_opt.keys()))

            if est_opt_keys_sorted != sub_cat:
                raise ValueError("estimator options dictionaries sub-catagories {} do not match required sub-catagories based on estimator string {}!".format(est_opt_keys_sorted, sub_cat))
            if est_opt["run_options"] is None:
                est_opt["run_options"] = {}
            if not isinstance(est_opt["run_options"], Dict):
                raise ValueError("run options must be a dictionary!")
            
            
        elif est_prim_str == "ion_trap":
            sub_cat = ["backend_access_path", "run_options", "transpilation_options", "abelian_grouping", "bound_pass_manager", "skip_transpilation"]
            sub_cat.sort()

            est_opt_keys_sorted = sorted(list(est_opt.keys()))

            if est_opt_keys_sorted != sub_cat:
                raise ValueError("estimator options dictionaries sub-catagories {} do not match required sub-catagories based on estimator string {}!".format(est_opt_keys_sorted, sub_cat))
            if est_opt["run_options"] is None:
                est_opt["run_options"] = {}
            if not isinstance(est_opt["run_options"], Dict):
                raise ValueError("run options must be a dictionary!")
            
            # if shots is not set, set it to default
            shots = est_opt["run_options"].get("shots", None)
            if shots is None:
                shots = 100
                print("number of shots is undefined. Set it to {} as default.".format(shots))
                est_opt["run_options"]["shots"] = shots
            
            if est_opt["transpilation_options"] is None:
                est_opt["transpilation_options"]={}
            if not isinstance(est_opt["transpilation_options"], Dict):
                raise ValueError("transpilation options must be a dictionary")
            if est_opt["bound_pass_manager"] is not None:
                if not isinstance(est_opt["bound_pass_manager"], PassManager):
                    raise ValueError("bound pass manager must be a qiskit.transpiler.PassManager object or None!")
            if not isinstance(est_opt["abelian_grouping"], bool):
                raise ValueError("abelian_grouping flag must be bool!")
            if not isinstance(est_opt["skip_transpilation"], bool):
                raise ValueError("skip_transpilation flag must be bool!")
            if not isinstance(est_opt["backend_access_path"], str):
                raise ValueError("Path to backend access file must be a string!")
            if not os.path.isfile(est_opt["backend_access_path"]):
                raise ValueError("File to load access data for backend not found! Searched for {}".format(est_opt["backend_access_path"]))
                
        else:
            raise ValueError("estimator string {} does not match any known string!".format(est_prim_str))

        return est_opt
    
    def get_filevector(self) -> Tuple[List, List]:
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
            circ_opt_lvl = self.estimator_options["transpilation_options"].get("optimization_level")
            shots = self.estimator_options["run_options"].get("shots", None)
            if shots is None:
                shots = self.estimator_options["backend_options"].get("shots", 0)
            abelian_grouping = self.estimator_options["abelian_grouping"]
                
        elif self.estimator_str == "ibm_runtime":
            err_mitig_meth = self.estimator_options["resilience_level"]

            circ_opt_lvl = self.estimator_options["optimization_level"]
            
            shots = self.estimator_options["execution_options"].get("shots")
            # IBM estimator always uses abelian_grouping 
            # https://quantumcomputing.stackexchange.com/questions/34694/is-qiskits-estimator-primitive-running-paulistrings-in-parallel
            abelian_grouping = True

        elif self.estimator_str == "terra":
            err_mitig_meth = 0
            circ_opt_lvl = 0
            shots = self.estimator_options["run_options"].get("shots", 0)

        elif self.estimator_str == "ion_trap":
            err_mitig_meth = 0
            circ_opt_lvl = self.estimator_options["transpilation_options"].get("optimization_level", None)
            if circ_opt_lvl is None:
                circ_opt_lvl = 0
            shots = self.estimator_options["run_options"].get("shots", 0)
            abelian_grouping = self.estimator_options["abelian_grouping"]
        
        header.append("err_mitigation")
        data.append(err_mitig_meth)

        header.append("circ_optimization")
        data.append(circ_opt_lvl)

        header.append("meas_shots")
        data.append(shots)
        
        header.append("backend_str")
        data.append(self.backend_str)
        
        header.append("noise_model")
        data.append(self.noise_model_str)
        
        header.append("coupling_map")
        data.append(self.coupling_map_str)

        header.append("basis_gates")
        data.append(self.basis_gates_str)

        header.append("abelian_grouping")
        data.append(abelian_grouping)
        

        return header, data
    
def get_EstimatorCalibration_from_dict(est_cal_dict: dict) -> EstimatorCalibration:

    est_opt = est_cal_dict.pop("estimator_options", None)
    if est_opt is None:
        raise ValueError("could not retrieve estimator options!")
    # check if noise_model is either None or a valid qiskit aer NoiseModel object
    noise_model_str = est_cal_dict.pop("noise_model_str", None)
    if noise_model_str is None:
        raise ValueError("could not retrieve noise model string from file!")
    
    valid_noise_model_found = False
    for key in est_opt.keys():
        if isinstance(est_opt[key], Dict):
            noise_model = est_opt[key].get("noise_model", None)
            if noise_model is not None:
                if not isinstance(noise_model, NoiseModel):
                    raise ValueError("Loaded noise model is no qiskit_aer NoiseModel!")
                if noise_model_str == "None":
                    raise ValueError("Noise model is a valid aer NoiseModel but noise model string equals string None!")
                valid_noise_model_found = True

    if not valid_noise_model_found:
        if noise_model_str != "None":
            raise ValueError("Noise model is None but noise model string {} does not match string None.".format(noise_model_str))
        
    coupling_map_str = est_cal_dict.pop("coupling_map_str", None)
    if coupling_map_str is None:
        raise ValueError("could not retrieve coupling map string from file!")
    basis_gates_str = est_cal_dict.pop("basis_gates_str", None)
    if basis_gates_str is None:
        raise ValueError("could not retrieve basis gates string from file!")
    
    est_prim_str = est_cal_dict.pop("estimator_str", None)
    if est_prim_str is None:
        raise ValueError("could not retrieve estimator string from file!")
    backend_str = est_cal_dict.pop("backend_str", None)
    if backend_str is None:
        raise ValueError("could not retrieve backend string from file!")

    name = est_cal_dict.pop("name", None)

    est_cal = EstimatorCalibration(est_opt, noise_model_str, coupling_map_str, basis_gates_str, est_prim_str, backend_str)
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
    

    est_opt = est_cal_dict.get("estimator_options", None)
    if est_opt is None:
        raise ValueError("could not retrieve estimator options!")
    
    # load possible noise_model
    for key in est_opt.keys():
        if isinstance(est_opt[key], Dict):
            fname_noise_model = est_opt[key].get("noise_model", None)
            if fname_noise_model is not None:
                
                if not os.path.isfile(fname_noise_model):
                    raise ValueError("Unable to find pickle file to load noise_model for estimator option {}. Looked for file {}.".format(key, fname_noise_model))
                
                with open(fname_noise_model, "rb") as f:
                    noise_model = pickle.load(f)

                if not isinstance(noise_model, NoiseModel):
                    raise ValueError("Loaded noise model is no qiskit_aer NoiseModel!")
                
                est_cal_dict["estimator_options"][key]["noise_model"] = noise_model

    return get_EstimatorCalibration_from_dict(est_cal_dict)
    

def get_EstimatorCalibration_from_pickle(fname: str) -> EstimatorCalibration:
    if not os.path.isfile(fname):
        raise ValueError("file {} does not exist!".format(fname))

    est_cal = None
    with open(fname, "rb") as f:
        est_cal = pickle.load(f)

    if not isinstance(est_cal, EstimatorCalibration):
        raise ValueError("loaded pickle object is no EstimatorCalibration!")

    return est_cal

def get_estimator(est_cal: EstimatorCalibration,
                  session: Union[qir.Session, None] = None) -> BaseEstimator:
    options_dict = est_cal.estimator_options
    if est_cal.estimator_str == "aer":
        est = AerEstimator(backend_options=options_dict["backend_options"], transpile_options=options_dict["transpilation_options"], run_options=options_dict["run_options"], approximation=options_dict["approximation"], skip_transpilation=options_dict["skip_transpilation"], abelian_grouping=options_dict["abelian_grouping"])
    elif est_cal.estimator_str == "ibm_runtime":
        options = qir.options.Options(optimization_level=options_dict["optimization_level"], resilience_level=options_dict["resilience_level"], max_execution_time=options_dict["max_execution_time"], transpilation=options_dict["transpilation_options"], resilience=options_dict["resilience_options"], execution=options_dict["execution_options"], environment=options_dict["environment_options"], simulator=options_dict["simulator_options"])
        est = qir.Estimator(session=session, options=options)
    elif est_cal.estimator_str == "terra":
        est = TerraEstimator(options=options_dict["run_options"])
    elif est_cal.estimator_str == "ion_trap":
        # with open(options_dict["backend_access_path"], 'r') as f:
        #     user = f.readline().strip()
        #     pw = f.readline().strip()
# 
        # backend = None
        # if self._parameters.backend_str == "umz_simulator":
        #     backend = UmzSimulatorBackend(email=user, password=pw)
        #     #backend = UmzSimulatorBackend()
        # elif self._parameters.backend_str == "red_trap":
        #     # setup RedTrapBackend
        #     backend = RedTrapBackend(email=user, password=pw)
        # else:
        #     raise ValueError("Backend string did not match any expected string!")
        # 
        # est = BackendEstimator(backend=backend, options=options_dict["run_options"], abelian_grouping=options_dict["abelian_grouping"], bound_pass_manager=options_dict["bound_pass_manager"], skip_transpilation=options_dict["skip_transpilation"])
        # 
        # if options_dict["transpilation_options"] is not None:
        # 
        #     est.set_transpile_options(**options_dict["transpilation_options"])
        raise NotImplementedError

    else:
        raise ValueError("estimator string {} in parameters does not match any known string!".format(est_cal.estimator_str))
    return est
        

class PresetPassManagerCalibration(Calibration):
    def __init__(self,
                 backend_str: str,
                 optimization_level: int = 0,
                 initial_layout: Union[List[int], None]=None,
                 layout_method: Union[str, None] = None,
                 routing_method: Union[str, None] = None,
                 translation_method: Union[str, None] = None,
                 scheduling_method: Union[str, None] = None,
                 approximation_degree: float = 1.0,
                 seed_transpile: Union[int, None] = None,
                 unitary_synthesis_method: str = 'default',
                 unitary_synthesis_plugin_config: Union[dict, None] = None):
        super().__init__("PresetPassManagerCalibration")
        
        self._validate_inputs(optimization_level, layout_method, routing_method, translation_method, scheduling_method, approximation_degree)
        
        self.optimization_level = optimization_level
        self.backend_str = backend_str
        self.initial_layout = initial_layout
        self.layout_method = layout_method
        self.routing_method = routing_method
        self.translation_method = translation_method
        self.scheduling_method = scheduling_method
        self.approximation_degree = approximation_degree
        self.seed_transpile = seed_transpile
        self.unitary_synthesis_method = unitary_synthesis_method
        self.unitary_synthesis_plugin_config = unitary_synthesis_plugin_config
    
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
            if k != "name":
                header.append(k)
                data.append(v)
        return (header, data)
    
    def _validate_inputs(self,
                         optimization_level: int = 0,
                         layout_method: Union[str, None] = None,
                         routing_method: Union[str, None] = None,
                         translation_method: Union[str, None] = None,
                         scheduling_method: Union[str, None] = None,
                         approximation_degree: float = 1.0):
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
        
            
def get_PresetPassManagerCalibration_from_dict(cal_dict: dict) -> PresetPassManagerCalibration:

    opt_lvl = cal_dict.get("optimization_level", None)
    if opt_lvl is None:
        print("Could not retrieve optimization level! Set it to default value 0")
        opt_lvl = 0

    backend_str = cal_dict.get("backend_str", None)
    if backend_str is None:
        raise ValueError("Could not retrieve backend string from dictionary!")

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

    return PresetPassManagerCalibration(backend_str, 
                                        optimization_level=opt_lvl, 
                                        initial_layout=initial_layout, 
                                        layout_method=layout_method, 
                                        routing_method=routing_method, 
                                        translation_method=translation_method,
                                        scheduling_method=scheduling_method, 
                                        approximation_degree=approximation_degree,
                                        seed_transpile=seed_transpile,
                                        unitary_synthesis_method=unitary_synthesis_method,
                                        unitary_synthesis_plugin_config=unitary_synthesis_plugin_config)

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
    
    
    
    return get_EstimatorCalibration_from_dict(pm_cal_dict)
    

def get_PresetPassManagerCalibration_from_pickle(fname: str) -> PresetPassManagerCalibration:
    if not os.path.isfile(fname):
        raise ValueError("file {} does not exist!".format(fname))

    pm_cal = None
    with open(fname, "rb") as f:
        pm_cal = pickle.load(f)

    if not isinstance(pm_cal, PresetPassManagerCalibration):
        raise ValueError("loaded pickle object is no PresetPassManagerCalibration!")

    return pm_cal

## To-Do: This function still needs to be implemented
def get_passmanager(pm_cal: PresetPassManagerCalibration) -> PassManager:
    # load IBM Quantum credentials
    load_dotenv()
    ibmq_api_token = os.environ["TOKEN"]
    ibmq_hub = os.environ["HUB"]
    ibmq_group = os.environ["GROUP"]
    ibmq_project = "multiflavorschwi"
    # load backend from backend string
    provider = IBMProvider(token=ibmq_api_token, instance=f"{ibmq_hub}/{ibmq_group}/{ibmq_project}")
    backend = provider.get_backend(pm_cal.backend_str)
    #raise NotImplementedError
    return generate_preset_pass_manager(optimization_level=pm_cal.optimization_level, 
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
##