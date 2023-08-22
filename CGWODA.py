import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from zoofs.baseoptimizationalgorithm import BaseOptimizationAlgorithm
import random

class HybridOptimizationAlgorithm(BaseOptimizationAlgorithm):
    def __init__(
        self,
        objective_function,
        n_iteration: int = 1000,
        timeout: int = None,
        population_size=50,
        # method=1,
        minimize=True,
        logger=None,
        **kwargs,
    ):
        super().__init__(
            objective_function, n_iteration, timeout, population_size, minimize, logger, **kwargs
        )
        # self.method = method
        


    def _evaluate_fitness(
        self, model, x_train, y_train, x_valid, y_valid, particle_swarm_flag=0, dragon_fly_flag=0
    ):
        return super()._evaluate_fitness(
            model, x_train, y_train, x_valid, y_valid, particle_swarm_flag, dragon_fly_flag
        )

    # def _check_params(self, model, x_train, y_train, x_valid, y_valid, method):
    #     super()._check_params(model, x_train, y_train, x_valid, y_valid)
        
    def gauss_map(self, x, a):
        return a * x * np.exp(-x**2)
   
          
    def fit(self, model, X_train, y_train, X_test, y_test, verbose=True):
        
        k=0
        kmax =2
        x = 0.5
        a_value=0.4
        self._check_params(model, X_train, y_train, X_test, y_test)

        self.feature_score_hash = {}
        kbest = self.population_size - 1
        self.feature_list = np.array(list(X_train.columns))
        self.best_results_per_iteration = {}
        self.best_score = np.inf
        self.worst_score = -np.inf
        self.worst_dim = np.ones(X_train.shape[1])
        self.best_dim = np.ones(X_train.shape[1])

        self.best_score_dimension = np.ones(X_train.shape[1])
        delta_x = np.random.randint(0, 2, size=(self.population_size, X_train.shape[1]))
        self.local_best_scores=[]
        self.initialize_population(X_train)

        self.alpha_wolf_dimension, self.alpha_wolf_fitness = np.ones(X_train.shape[1]), np.inf
        self.beta_wolf_dimension, self.beta_wolf_fitness = np.ones(X_train.shape[1]), np.inf
        self.delta_wolf_dimension, self.delta_wolf_fitness = np.ones(X_train.shape[1]), np.inf
        x = 0.5 
        

        for i in range(self.n_iteration):
            self._check_individuals()
            if k <= kmax/2:
                #dragonfly algorithm using 'An improved Dragonfly Algorithm for feature selection'
                
                
                self.fitness_scores = self._evaluate_fitness(
                    model, X_train, y_train, X_test, y_test, 0, 1
                )

                self.iteration_objective_score_monitor(i)

                
                #Equation 7
                if 2 * (i + 1) <= self.n_iteration:
                    #Equation 9
                    pct = 0.1 - (0.2 * (i + 1) / self.n_iteration)
                else:
                    pct = 0

                #Equation 8
                w = 0.9 - (i + 1) * (0.5) / (self.n_iteration)
                s = 2 * np.random.random() * pct
                a = 2 * np.random.random() * pct
                c = 2 * np.random.random() * pct
                f = 2 * np.random.random()
                e = pct

                
                    
                temp = individuals = self.individuals
                temp_2 = (
                    temp.reshape(temp.shape[0], 1, temp.shape[1])
                    - temp.reshape(1, temp.shape[0], temp.shape[1])
                ).reshape(temp.shape[0] ** 2, temp.shape[1]) ** 2
                temp_3 = temp_2.reshape(temp.shape[0], temp.shape[0], temp.shape[1]).sum(axis=2)
                zz = np.argsort(temp_3)
                cc = [list(iter1[iter1 != iter2]) for iter1, iter2 in zip(zz, np.arange(temp.shape[0]))]

                si = -(
                    np.repeat(individuals, kbest, axis=0).reshape(
                        individuals.shape[0], kbest, individuals.shape[1]
                    )
                    - individuals[np.array(cc)[:, :kbest]]
                ).sum(axis=1)
               
                ai = delta_x[np.array(cc)[:, :kbest]].sum(axis=1) / kbest
                ci = (individuals[np.array(cc)[:, :kbest]].sum(axis=1) / kbest) - individuals
                fi = self.best_score_dimension - self.individuals
                ei = self.individuals + self.worst_dim
                
                #Equation 6
                delta_x = s * si + a * ai + c * ci + f * fi + e * ei + w * delta_x
                delta_x = np.where(delta_x > 6, 6, delta_x)
                delta_x = np.where(delta_x < -6, -6, delta_x)
                
                #Equation 11 and 12
                T = abs(delta_x / np.sqrt(1 + delta_x ** 2))
                #Update individual positions
                self.individuals = np.where(
                    np.random.uniform(size=(self.population_size, X_train.shape[1])) < T,
                    np.logical_not(self.individuals).astype(int),
                    individuals,
                )
                
                self.best_feature_list = list(self.feature_list[np.where(self.best_dim)[0]])
                

            else:
                if k == kmax:
                    chaotic_value = self.gauss_map(x, a_value)
                    x = chaotic_value
                    
                    delta_x = chaotic_value * si + chaotic_value * ai + chaotic_value * ci + chaotic_value * fi + chaotic_value * ei + chaotic_value * delta_x
                    delta_x = np.where(delta_x > 6, 6, delta_x)
                    delta_x = np.where(delta_x < -6, -6, delta_x)

                    #Equation 11 and 12
                    T = abs(delta_x / np.sqrt(1 + delta_x ** 2))
                    #Update individual positions
                    self.individuals = np.where(
                        np.random.uniform(size=(self.population_size, X_train.shape[1])) < T,
                        np.logical_not(self.individuals).astype(int),
                        individuals,
                    )

                  
                    self.best_feature_list = list(self.feature_list[np.where(self.best_dim)[0]])
                
                    k=0
                    
                    
                 #grey wolf         
                a_w = 2 - 2 * ((i + 1) / self.n_iteration)

                self.fitness_scores = self._evaluate_fitness(model, X_train, y_train, X_test, y_test)

                self.iteration_objective_score_monitor(i)

                top_three_fitness_indexes = np.argsort(self.fitness_scores)[:3]

                for fit, dim in zip(
                    np.array(self.fitness_scores)[top_three_fitness_indexes],
                    self.individuals[top_three_fitness_indexes],
                ):
                    if fit < self.alpha_wolf_fitness:
                        self.delta_wolf_fitness = self.beta_wolf_fitness
                        self.beta_wolf_fitness = self.alpha_wolf_fitness
                        self.alpha_wolf_fitness = fit

                        self.delta_wolf_dimension = self.beta_wolf_dimension
                        self.beta_wolf_dimension = self.alpha_wolf_dimension
                        self.alpha_wolf_dimension = dim
                        continue

                    if (fit > self.alpha_wolf_fitness) & (fit < self.beta_wolf_fitness):
                        self.delta_wolf_fitness = self.beta_wolf_fitness
                        self.beta_wolf_fitness = fit

                        self.delta_wolf_dimension = self.beta_wolf_dimension
                        self.beta_wolf_dimension = dim
                        continue

                    if (fit > self.beta_wolf_fitness) & (fit < self.delta_wolf_fitness):
                        self.delta_wolf_fitness = fit
                        self.delta_wolf_dimension = dim

                
                C1 = 2 * np.random.random((self.population_size, X_train.shape[1]))
                A1 = 2 * a_w * np.random.random((self.population_size, X_train.shape[1])) - a_w
                d_alpha = abs(C1 * self.alpha_wolf_dimension - self.individuals)

                C2 = 2 * np.random.random((self.population_size, X_train.shape[1]))
                A2 = 2 * a_w * np.random.random((self.population_size, X_train.shape[1])) - a_w
                d_beta = abs(C2 * self.beta_wolf_dimension - self.individuals)

                C3 = 2 * np.random.random((self.population_size, X_train.shape[1]))
                A3 = 2 * a_w * np.random.random((self.population_size, X_train.shape[1])) - a_w
                d_delta = abs(C3 * self.delta_wolf_dimension - self.individuals)

                
                X1 = abs(self.alpha_wolf_dimension - A1 * d_alpha)
                X2 = abs(self.beta_wolf_dimension - A2 * d_beta)
                X3 = abs(self.delta_wolf_dimension - A3 * d_delta)
                self.individuals = np.where(
                    np.random.uniform(size=(self.population_size, X_train.shape[1]))
                    <= self.sigmoid((X1 + X2 + X3) / 3),
                    1,
                    0,
                )

                
                
                self.best_feature_list = list(self.feature_list[np.where(self.worst_dim == max(self.worst_dim))])
           
            if i > 2 and self.best_score == self.best_results_per_iteration[i-1]['best_score']:
                k += 1
            else:
                k = 0
            self.verbose_results(verbose, i)
        

        return self.best_feature_list
    
    
