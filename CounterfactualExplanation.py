import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from config import cfg
from AMPLSolver import AMPLSolver


class CE(AMPLSolver):
    def __init__(self, classifier, dataset, log_file):
        super().__init__()
        self.classifier = classifier
        self.dataset = dataset
        self.log_file = log_file
        self.ampl_model_path = "ATM_CE.mod"
        self.F = self.dataset.x_tr.shape[1]

        self.timelim = cfg.CE_timelim
        self.ampl_model = self.build_ampl_model(self.ampl_model_path, self.gurobi)
        self.set_model_parameters()

    # Objective: look for the counterfactual explanation of x0 with class k
    def solve(self, x0, k_star, initialPoint=None):

        t0 = time.time()
        self.k_star = k_star
        self.x0 = x0
        self.initialPoint = initialPoint

        self.update_x0()

        if self.initialPoint is not None:
            self.init_solution(self.initialPoint)

        super().solve()

        x = self.ampl_model.getVariable("x").getValues()
        self.x = np.array([x.getRow((v + 1))[1] for v in range(len(self.x0))]).reshape((1, -1))

        total_exec_time = time.time() - t0
        print(f"{total_exec_time}", file=open(cfg.get_counterfactual_log(),'a'))
        return self.x

    def init_solution(self, x):
        start_time = time.time()
        self.ampl_model.eval("objective none;")
        var_ = self.ampl_model.getVariable("x")
        self.set_1d_parameter_value(var_, x, "x", var_.setValues)
        self.ampl_model.eval("fix x;")
        start_time = time.time()
        self.ampl_model.solve()
        self.ampl_model.eval("objective C;")
        self.ampl_model.eval("unfix x;")

        self.ampl_model.setOption("gurobi_options", "timelim {}".format(self.timelim))
        print(f"Time for initializing CE solution {time.time()-start_time} seconds")

    def l1_distance(self, x1, x2):
        return np.linalg.norm(x1-x2, axis=1, ord=1)

    def l0_distance(self, x1, x2):
        return np.linalg.norm(np.round(x1,5)-np.round(x2,5), axis=1, ord=0)

    def set_model_parameters(self):
        super().set_model_parameters()
        self.set_SET_value("F", [i+1 for i in range(self.F)])
        self.set_SET_value("N", [i+1 for i in range(self.F) if self.dataset.features_type[i]=="N"])
        self.set_SET_value("D", [i+1 for i in range(self.F) if self.dataset.features_type[i]=="D"])
        self.set_SET_value("B", [i+1 for i in range(self.F) if self.dataset.features_type[i]=="B" or "C" in self.dataset.features_type[i]])

        self.set_parameter_value("c_onehot", self.dataset.c_onehot)

        for j in range(self.dataset.c_onehot):
            self.set_SET_value("C_OneHot", [i+1 for i in range(self.F) if self.dataset.features_type[i] == "C{}".format(j)], j+1)

        self.set_SET_value("A_INC", [i+1 for i in range(self.F) if self.dataset.features_actionability[i]=="INC"])
        self.set_SET_value("A_DEC", [i+1 for i in range(self.F) if self.dataset.features_actionability[i]=="DEC"])
        self.set_SET_value("A_FIX", [i+1 for i in range(self.F) if self.dataset.features_actionability[i]=="FIX"])

        self.set_SET_value("UB", [i+1 for i in range(self.F)])
        self.set_SET_value("LB", [i+1 for i in range(self.F)])

        self.set_1d_parameter_value(self.ampl_model.getParameter("ub"), self.dataset.ub, "ub", self.ampl_model.setData)
        self.set_1d_parameter_value(self.ampl_model.getParameter("lb"), self.dataset.lb, "lb", self.ampl_model.setData)

        self.set_parameter_value("lambda0", cfg.lambda0)
        self.set_parameter_value("lambda1", cfg.lambda1)
        self.set_parameter_value("lambda2", cfg.lambda2)

        self.set_SET_value("K", self.classifier.classes_)


        if isinstance(self.classifier, DecisionTreeClassifier):
            decision_trees = [self.classifier]
        elif isinstance(self.classifier, RandomForestClassifier):
            decision_trees = self.classifier.estimators_
        else:
            raise Exception('Unrecognized model in CounterfactualExplanation problem. Try with DecisionTreeClassifier or RandomForestClassifier')


        T = len(decision_trees)
        self.set_parameter_value("T",T)

        internal_nodes_list = []
        v_list = []
        c_list = []

        left_index0 = []
        left_index1 = []
        left_index2 = []
        left_list = []
        right_list = []

        w_index0 = []
        w_index1 = []
        w_index2 = []
        w_list = []
        for t in range(T):
            children_left = decision_trees[t].tree_.children_left.tolist()
            children_right = decision_trees[t].tree_.children_right.tolist()

            internal_nodes = np.where(decision_trees[t].tree_.feature >= 0)[0]
            self.set_SET_value("I_N", internal_nodes, int(t+1))

            leaves = np.where(decision_trees[t].tree_.feature < 0)[0]
            self.set_SET_value("L", leaves, int(t+1))

            node_labels = np.argmax(decision_trees[t].tree_.value, axis=2).reshape(-1, )
            node_labels = [self.classifier.classes_[i] for i in node_labels]

            for label_index in range(len(self.classifier.classes_)):
                label = self.classifier.classes_[label_index]
                leaves_labels = [i for i in range(len(node_labels)) if i in leaves and node_labels[i] == label]
                self.set_SET_value("Lk", leaves_labels, (int(t+1),int(label)))

                for leaf in leaves:
                    w_index0.append(t+1)
                    w_index1.append(leaf)
                    w_index2.append(label)
                    w_list.append(decision_trees[t].tree_.value[leaf,0,label_index]/np.sum(decision_trees[t].tree_.value[leaf,0,:]))

            for leaf in leaves:
                ancestors = []
                left_ancestors = []
                right_ancestors = []
                ll = leaf
                while True:
                    if ll in children_left:
                        ll = children_left.index(ll)
                        left_ancestors.append(ll)
                    elif ll in children_right:
                        ll = children_right.index(ll)
                        right_ancestors.append(ll)
                    else:
                        break
                    ancestors.append(ll)

                for n in internal_nodes:
                    left_index0.append(t+1)
                    left_index1.append(leaf)
                    left_index2.append(n)
                    left_list.append(n in left_ancestors)
                    right_list.append(n in right_ancestors)

            internal_nodes_list.append(internal_nodes)
            v_list.append(decision_trees[t].tree_.feature[decision_trees[t].tree_.feature >= 0]+1)
            c_list.append(decision_trees[t].tree_.threshold[decision_trees[t].tree_.threshold >= 0])

        self.set_Nd_parameter_value(self.ampl_model.getParameter("Left"), [left_index0, left_index1, left_index2], left_list, "Left", self.ampl_model.setData)
        self.set_Nd_parameter_value(self.ampl_model.getParameter("Right"), [left_index0, left_index1, left_index2], right_list, "Right", self.ampl_model.setData)

        self.set_1d_parameter_value(self.ampl_model.getParameter("eps"), self.dataset.eps, "eps", self.ampl_model.setData)

        self.set_Nd_parameter_value(self.ampl_model.getParameter("v"), [[i+1 for i in range(T) for j in range(len(internal_nodes_list[i]))], [internal_nodes_list[i][j] for i in range(T) for j in range(len(internal_nodes_list[i]))]], [v_list[i][j] for i in range(T) for j in range(len(v_list[i]))], "v", self.ampl_model.setData)
        self.set_Nd_parameter_value(self.ampl_model.getParameter("c"), [[i+1 for i in range(T) for j in range(len(internal_nodes_list[i]))], [internal_nodes_list[i][j] for i in range(T) for j in range(len(internal_nodes_list[i]))]], [c_list[i][j] for i in range(T) for j in range(len(c_list[i]))], "c", self.ampl_model.setData)

        self.set_Nd_parameter_value(self.ampl_model.getParameter("w"), [w_index0, w_index1, w_index2], w_list, "w", self.ampl_model.setData)

    def update_x0(self):
        print("Updating x0 parameters")
        self.ampl_model.eval('update data x0,k_star;')
        par_ = self.ampl_model.getParameter("x0")
        self.set_1d_parameter_value(self.ampl_model.getParameter("x0"), self.x0, "x0", par_.setValues)
        self.set_parameter_value("k_star", self.classifier.classes_.tolist().index(self.k_star))