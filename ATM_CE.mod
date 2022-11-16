reset;

param T>=1; # number of trees
set F; # set of features

set N within F; # set of numerical features
set D within F; # set of discrete features
set B within F; # set of binary features

param c_onehot>=0; # number of categorical features
set C_OneHot{1..c_onehot} within B; # set of categorical features

set A_INC within F; # set of actionability: only increments are allowed
set A_DEC within F; # set of actionability: only decrements are allowed
set A_FIX within F; # set of actionability: no changes are allowed

set UB within F; # set of upper bounded features
param ub{UB}; # set of upper bounds
set LB within F; # set of lower bounded features
param lb{LB}; # set of lower bounds

set K; # set of labels

param x0{j in F}; # initial point
param k_star in K; # required class for the counterfactual explanation

set I_N{t in 1..T}; # set of internal nodes for tree t
set L{t in 1..T}; # set of leaf for tree t
set NL{t in 1..T} := L[t] union I_N[t];
set Lk{t in 1..T, k in K} within L[t]; # 1 if n is leaf for tree t with label k
param Left{t in 1..T, l in L[t], n in I_N[t]} binary; # set of ancestors for leaf l in tree t whose
								# left branch takes part in the path that ends in l
param Right{t in 1..T, l in L[t], n in I_N[t]} binary; # set of ancestors for leaf l in tree t whose
								# right branch takes part in the path that ends in l
						
param v{t in 1..T, s in I_N[t]} in F; # index of the feature used in split s for tree t
param c{t in 1..T, s in I_N[t]}; # threshold used in split s for tree t
param w{t in 1..T, l in L[t], k in K}; # weight for tree t

var x{j in F}; # counterfactual
var xN{j in N};
var xD{j in D};
var xB{j in B} binary;
var z{t in 1..T, l in L[t]} binary; # 1 if counterfactual x ends in leaf l for tree t
var x_l0{j in F} binary;
var x_l1{j in F}>=0,<=1;

param M1 := 1.5;
param M2 := 1;
param M3 := 1;
param lambda0 >= 0;
param lambda1 >= 0;
param lambda2 >= 0;
param eps{j in F};# = 0.005;

minimize C: lambda0*sum{j in F}x_l0[j] + lambda1*sum{j in F}x_l1[j] + lambda2*sum{j in F}(x0[j]-x[j])^2;
minimize none: 0;

s.t. numeric_features{j in N}: x[j] = xN[j];
s.t. discrete_features{j in D}: x[j] = xD[j];
s.t. binary_features{j in B}: x[j] = xB[j];

s.t. L0_constr1{j in F}: x0[j]-x[j] >= -M3*x_l0[j];
s.t. L0_constr2{j in F}: x0[j]-x[j] <= M3*x_l0[j];

s.t. L1_constr1{j in F}: x_l1[j] >= x0[j]-x[j];
s.t. L1_constr2{j in F}: x_l1[j] >= -x0[j]+x[j];

s.t. branch_left{t in 1..T, l in L[t], s in I_N[t]: Left[t,l,s]==1}: x[v[t,s]] - M1*(1-z[t,l]) + eps[v[t,s]] <= c[t,s];
s.t. branch_right{t in 1..T, l in L[t], s in I_N[t]: Right[t,l,s]==1}: x[v[t,s]] + M2*(1-z[t,l]) - eps[v[t,s]] >= c[t,s];

s.t. leaf_assignment{t in 1..T}: sum{l in L[t]}z[t,l]=1;

s.t. class_assignment{k in K: k!=k_star}: sum{t in 1..T, l in L[t]} 1/T*w[t,l,k_star]*z[t,l] >= 0.0001 + sum{t in 1..T, l in L[t]} 1/T*w[t,l,k]*z[t,l];

s.t. actionability_inc{j in A_INC}: x[j] >= x0[j];
s.t. actionability_dec{j in A_DEC}: x[j] <= x0[j];
s.t. actionability_fix{j in A_FIX}: x[j] = x0[j];

s.t. type_onehot{i in 1..c_onehot}: sum{j in C_OneHot[i]}x[j]=1;

s.t. type_ub{j in UB}: x[j]<=ub[j];
s.t. type_lb{j in LB}: x[j]>=lb[j];

s.t. admissibility: sum{j in F}x_l0[j]>=1;