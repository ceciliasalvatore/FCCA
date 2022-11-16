import amplpy
import numpy as np
import pandas as pd
from amplpy import AMPL, Environment, DataFrame

import config as cfg
import time

class AMPLSolver:
    def __init__(self):
        self.ampl_model = None
        self.gurobi = "gurobi"
        self.knitro = "knitro"
        self.cplex = "cplex"
        self.ampl_model_path = None
        self.timelim = None

    def solve(self):
        # Solve AMPL model
        start_time = time.time()
        self.ampl_model.solve()
        exec_time = time.time()-start_time
        print("AMPL Solving Time: {}".format(time.strftime("%H:%M:%S", time.gmtime(exec_time))))

    def build_ampl_model(self, model_path, solver):
        print("Building AMPL Problem")
        # Initialize AMPL
        ampl = AMPL(environment=Environment(cfg.ampldir))
        # Load .mod file
        ampl.read(model_path)
        # Set solver
        ampl.setOption("solver", solver)
        if self.timelim is not None:
            print("timelim {}".format(self.timelim))
            ampl.setOption("gurobi_options", "timelim {}".format(self.timelim))
            ampl.setOption("cplex_options", "time {}".format(self.timelim))
        else:
            ampl.setOption("gurobi_options","outlev 1")
        ampl.setOption("solver_msg", "1")
        ampl.setOption("substout", "1")

        return ampl

    def set_model_parameters(self):
        print("Setting AMPL parameters")
        pass;

    def set_parameter_value(self, name, value):
        self.ampl_model.getParameter(name).set(value)

    def set_SET_value(self, name, list, index=None):
        set = self.ampl_model.getSet(name)
        if index != None:
            set = set[index]
        set.setValues(list)

    def set_Nd_parameter_value(self, par_, index_list, value, name, setFunction):
        par_index = par_.getIndexingSets()
        assert len(par_index) == len(index_list)
        par_data = {}
        for i in range(len(par_index)):
            par_data[par_index[i]] = index_list[i]
        par_data[name] = value
        par_df = pd.DataFrame(par_data).set_index(par_index)
        setFunction(DataFrame.fromPandas(par_df, par_index))

    def set_1d_parameter_value(self, par_, value, name, setFunction):
        par_index = par_.getIndexingSets()
        par_data = {par_index[0]: [i for i in range(1, len(value) + 1)], name: value}
        par_df = pd.DataFrame(par_data).set_index(par_index)
        setFunction(DataFrame.fromPandas(par_df, par_index))