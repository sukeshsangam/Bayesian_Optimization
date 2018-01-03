from scipy.optimize import differential_evolution
import math
import GPyOpt
import GPy
import numpy as np
from numpy.random import seed # fixed seed
from sklearn.ensemble import RandomForestRegressor
import math
seed(123456)
func = GPyOpt.objective_examples.experiments2d.sixhumpcamel()
space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-100,100)},
        {'name': 'var_2', 'type': 'continuous', 'domain': (-100,100)}]
'''
		constrains = [{'name': 'constr_1', 'constrain': '-x[:,1] -.5 + abs(x[:,0]) - np.sqrt(1-x[:,0]**2)'},
              {'name': 'constr_2', 'constrain': 'x[:,1] +.5 - abs(x[:,0]) - np.sqrt(1-x[:,0]**2)'}]
			  '''
feasible_region = GPyOpt.Design_space(space = space)


initial_design = GPyOpt.util.stats.initial_design('random', feasible_region, 10)

def beale(x):
	#print(x)
	x1 = x[0]
	x11 = x1[0]
	x22 = x1[1]
	
	


	fact1 = -math.cos(x11)*math.cos(x22);
	fact2 = math.exp(-(x11-math.pi)**2-(x22-math.pi)**2);

	y = fact1*fact2;


	return y


	
	
def beale_f(x):
	#x.reshape(1,2)
	#print(x)
	x1 = x[0]
	#print(x1)
	#x11 = x1[0]
	x2=x[1]

	


	fact1 = -math.cos(x1)*math.cos(x2);
	fact2 = math.exp(-(x1-math.pi)**2-(x2-math.pi)**2);

	y = fact1*fact2;


	return y

	
# --- CHOOSE the objective
objective = GPyOpt.core.task.SingleObjective(beale)

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
print(beale_f(bo.x_opt))

print(bo.get_evaluations())


bounds = [(-100, 100), (-100,100)]
result = differential_evolution(beale_f, bounds)
print(result.x)
print(result.fun)
print(result.nfev)