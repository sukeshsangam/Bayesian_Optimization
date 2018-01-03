
import GPyOpt
import GPy
import numpy as np
from numpy.random import seed # fixed seed
from sklearn.ensemble import RandomForestRegressor
seed(123456)
func = GPyOpt.objective_examples.experiments2d.sixhumpcamel()
space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-1,1)},
        {'name': 'var_2', 'type': 'continuous', 'domain': (-1.5,1.5)}]
constrains = [{'name': 'constr_1', 'constrain': '-x[:,1] -.5 + abs(x[:,0]) - np.sqrt(1-x[:,0]**2)'},
              {'name': 'constr_2', 'constrain': 'x[:,1] +.5 - abs(x[:,0]) - np.sqrt(1-x[:,0]**2)'}]
feasible_region = GPyOpt.Design_space(space = space, constraints = constrains)


initial_design = GPyOpt.util.stats.initial_design('random', feasible_region, 10)

def six_hump(x):
	x.reshape(1,2)
	print(x)
	term1 = (4-2.1*x[0]**2+(x[0]**4)/3) * x[0]**2
	term2 = x[0]*x[1]
	term3 = (-4+4*x[1]**2) * x[1]**2
	fval = term1 + term2 + term3
	return fval
# --- CHOOSE the objective
objective = GPyOpt.core.task.SingleObjective(func.f)

# --- CHOOSE the model type
model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=10,verbose=False)

# --- CHOOSE the acquisition optimizer
aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

# --- CHOOSE the type of acquisition
acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

# --- CHOOSE a collection method
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

# BO object
bo = GPyOpt.methods.ModularBayesianOptimization(model, space, objective, acquisition, evaluator, initial_design)

max_iter  = 25
max_time  = None 
tolerance = 1e-8
bo.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=False)
#bo.save_models( models_file='report.txt')
#print(bo.get_evaluations())

print(bo.x_opt)
#print(func)
print(six_hump(bo.x_opt))

def msixhumpcamel(x):
    z = np.zeros((1,2))
    z[0,:] = x
    y_pred = rf.predict(z)
    return y_pred
n = 3600
x = 10*np.random.rand(n,2)-1
#print(x)
y = np.zeros(n)
for i in range(0,n):
    y[i] = func.f(x[i,:])
rf = RandomForestRegressor()
#print(x)
#print(y)
rf.fit(x, y)
objective_m = GPyOpt.core.task.SingleObjective(msixhumpcamel)
bo_m = GPyOpt.methods.ModularBayesianOptimization(model, space, objective_m, acquisition, evaluator, initial_design)
bo_m.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=False)


print(bo_m.x_opt)
print(six_hump(bo_m.x_opt))
#print(bo.X)
print(rf.predict((bo.x_opt).reshape(1,-1)))


x1=np.loadtxt('newfile1.txt')
numrows=len(x1)
y1 = np.zeros(numrows)
rf1 = RandomForestRegressor()
for i in range(0,numrows):
    y1[i] = six_hump(x1[i,:])
	
rf1.fit(x1, y1)
def msixhumpcamel1(x):
    z = np.zeros((1,2))
    z[0,:] = x
    y_pred = rf1.predict(z)
    return y_pred

objective_m1 = GPyOpt.core.task.SingleObjective(msixhumpcamel1)
bo_m1 = GPyOpt.methods.ModularBayesianOptimization(model, space, objective_m1, acquisition, evaluator, initial_design)
bo_m1.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=False)
print(bo_m1.x_opt)
print(six_hump(bo_m1.x_opt))
print(rf1.predict((bo.x_opt).reshape(1,-1)))