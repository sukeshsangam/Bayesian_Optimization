from scipy.optimize import differential_evolution
import math
import GPyOpt
import GPy
import numpy as np
from numpy.random import seed # fixed seed
from sklearn.ensemble import RandomForestRegressor
seed(123456)
func = GPyOpt.objective_examples.experiments2d.sixhumpcamel()
space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-2,2)},
        {'name': 'var_2', 'type': 'continuous', 'domain': (-2,2)}]
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
	
	fact1a = (x11 + x22 + 1)**2
	fact1b = 19 - 14*x11 + 3*x11**2 - 14*x22 + 6*x11*x22 + 3*x22**2
	fact1 = 1 + fact1a*fact1b
	
	fact2a = (2*x11 - 3*x22)**2
	fact2b = 18 - 32*x11 + 12*x11**2 + 48*x22 - 36*x11*x22 + 27*x22**2
	fact2 = 30 + fact2a*fact2b
	
	y =fact1*fact2
	return y


	
	
def beale_f(x):
	#x.reshape(1,2)
	#print(x)
	x1 = x[0]
	#print(x1)
	#x11 = x1[0]
	x2=x[1]

	fact1a = (x1 + x2 + 1)**2
	fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
	fact1 = 1 + fact1a*fact1b
	
	fact2a = (2*x1 - 3*x2)**2
	fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
	fact2 = 30 + fact2a*fact2b
	
	y =fact1*fact2
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


bounds = [(-2, 2), (-2, 2)]
result = differential_evolution(beale_f, bounds)
print(result.x)
print(result.fun)