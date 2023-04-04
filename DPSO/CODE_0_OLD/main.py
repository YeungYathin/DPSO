from input import getData
import numpy as np
import sys
import random
import time

f = None

class Main:

    # Basic attribute of the UFLP problem
    m = None
    n = None
    f = None
    c = None

    generation = None
    goal_generation = None
    particle_list = None
    gbest_facility_vector = None
    gbest_fitness_value = None
    gbest_index = None
    value_list = None
    
    start = None

    def __init__(self, goal_generation, datapath):
        self.read_data(datapath=datapath)
        Main.goal_generation = goal_generation

    def DPSO(self):
        self.initialize()
        while Main.generation < Main.goal_generation:
            print(f'{Main.generation} {Main.gbest_fitness_value} {np.mean(Main.value_list)} {np.var(Main.value_list)} {time.time() - Main.start}')
            print(f'{Main.generation} {Main.gbest_fitness_value} {np.mean(Main.value_list)} {np.var(Main.value_list)} {time.time() - Main.start}', file=f)
            Main.generation += 1
            self.update_pbest_gbest()
            Main.value_list = []
            for individual in Main.particle_list:
                individual.velocity()
                individual.cognition()
                individual.social()
                individual.calculate_fitness_value()
                Main.value_list.append(individual.fitness_value)
            self.local_search()
        return Main.gbest_facility_vector, Main.gbest_fitness_value

    def initialize(self):
        Main.start = time.time()
        Main.generation = 0
        Main.particle_list = []
        Main.value_list = []
        Main.gbest_fitness_value = int(sys.maxsize)
        for j in range(0, Main.m):
            initial_solution_particle = particle(j)
            Main.particle_list.append(initial_solution_particle)
            Main.value_list.append(initial_solution_particle.fitness_value)

    def update_pbest_gbest(self):
        for individual in Main.particle_list:
            if individual.fitness_value < Main.gbest_fitness_value:
                Main.gbest_fitness_value = individual.fitness_value
                Main.gbest_facility_vector = individual.facility_vector
                Main.gbest_index = individual.index_in_particle_list
            if individual.fitness_value < individual.pbest_fitness_value:
                individual.pbest_fitness_value = individual.fitness_value
                individual.pbest_facility_vector = individual.facility_vector

    def local_search(self):
        s0 = np.copy(Main.gbest_facility_vector)
        rand_facility_1_loc, rand_facility_2_loc = random.sample(
            range(0, Main.m), 2)
        s0[rand_facility_1_loc] = 1-s0[rand_facility_1_loc]
        s0[rand_facility_2_loc] = 1-s0[rand_facility_2_loc]
        s = s0
        fs = self.calculate_fitness_value(s)
        for l in range(0, Main.m):
            rand_facility_loc = random.randint(0, Main.m-1)
            s1 = np.copy(s)
            s1[rand_facility_loc] = 1-s1[rand_facility_loc]
            fs1 = self.calculate_fitness_value(s1)
            if fs1 < fs:
                s = s1
                fs = fs1
        if fs <= Main.gbest_fitness_value:
            Main.gbest_facility_vector = s
            Main.gbest_fitness_value = fs
            Main.particle_list[Main.gbest_index].pbest_fitness_value = fs

    def calculate_fitness_value(self, facility_vector):
        result = 0

        # Calculate the opening cost
        result += np.inner(facility_vector, Main.f)
        
        open_facility_indexes = []
        for j in range(0, Main.m):
            if facility_vector[j]:
                open_facility_indexes.append(j)

        # Calculate the facility-customer cost
        for i in range(0, Main.n):
            temp = int(sys.maxsize)
            for k in range(len(open_facility_indexes)):
                j = open_facility_indexes[k]
                if Main.c[i][j] < temp:
                    temp = Main.c[i][j]
            result += temp

        return result

    def read_data(self, datapath):
        Main.m, Main.n, Main.f, Main.c = getData(datapath=datapath)

    def test(self):  # This is just a test for correctness verify
        print(Main.m, Main.n, Main.f, Main.c)


class particle:

    c1 = 0.5  # social probability
    c2 = 0.5  # cognitive probability
    w = 0.9  # inertia weight

    def __init__(self, index_in_particle_list):
        self.facility_vector = None
        self.fitness_value = None
        self.pbest_facility_vector = None
        self.pbest_fitness_value = None
        self.index_in_particle_list = index_in_particle_list

        self.initialize()

    def initialize(self):
        # Initialize the facility vector with random 0/1
        self.facility_vector = np.random.randint(0, 2, Main.m)
        # Calculate the fitness value for the initial solution
        self.calculate_fitness_value()

        self.pbest_facility_vector = self.facility_vector
        self.pbest_fitness_value = self.fitness_value

    def calculate_fitness_value(self):
        result = 0

        # Calculate the opening cost
        result += np.inner(self.facility_vector, Main.f)

        open_facility_indexes = []
        for j in range(0, Main.m):
            if self.facility_vector[j]:
                open_facility_indexes.append(j)
        
        # Calculate the facility-customer cost
        for i in range(0, Main.n):
            temp = int(sys.maxsize)
            for k in range(len(open_facility_indexes)):
                j = open_facility_indexes[k]
                if Main.c[i][j] < temp:
                    temp = Main.c[i][j]
            result += temp

        self.fitness_value = result

    def velocity(self):
        r = random.random()
        if r < particle.w:
            rand_facility_1_loc, rand_facility_2_loc = random.sample(
                range(0, Main.m), 2)
            a = self.facility_vector[rand_facility_1_loc]
            b = self.facility_vector[rand_facility_2_loc]
            self.facility_vector[rand_facility_1_loc] = b
            self.facility_vector[rand_facility_2_loc] = a

    def cognition(self):
        r = random.random()
        if r < particle.c1:
            first_parant_lambda = np.copy(self.facility_vector)
            second_parant_pbest = np.copy(self.pbest_facility_vector)
            start_idx = 0
            end_idx = random.randint(1, Main.m-2)

            # One-cut crossover
            temp = np.copy(first_parant_lambda[start_idx:end_idx+1])
            first_parant_lambda[start_idx:end_idx +
                                1] = second_parant_pbest[start_idx:end_idx+1]
            second_parant_pbest[start_idx:end_idx+1] = temp

            self.facility_vector = random.choice(
                [first_parant_lambda, second_parant_pbest])

    def social(self):
        r = random.random()
        if r < particle.c2:
            first_parant_sigma = np.copy(self.facility_vector)
            second_parant_gbest = np.copy(Main.gbest_facility_vector)
            start_idx = random.randint(1, Main.m-2)
            end_idx = random.randint(start_idx+1, Main.m-1)

            # Two-cut crossover
            temp = np.copy(first_parant_sigma[start_idx:end_idx+1])
            first_parant_sigma[start_idx:end_idx +
                               1] = second_parant_gbest[start_idx:end_idx+1]
            second_parant_gbest[start_idx:end_idx+1] = temp

            self.facility_vector = random.choice(
                [first_parant_sigma, second_parant_gbest])


if __name__ == '__main__':
    datapath_list = ['/Users/YeungYathin/Desktop/创新实践/复现代码/PSO/M/O/MO1',
                     '/Users/YeungYathin/Desktop/创新实践/复现代码/PSO/M/O/MO2',
                     '/Users/YeungYathin/Desktop/创新实践/复现代码/PSO/M/O/MO3',
                     '/Users/YeungYathin/Desktop/创新实践/复现代码/PSO/M/O/MO4',
                     '/Users/YeungYathin/Desktop/创新实践/复现代码/PSO/M/O/MO5',
                     '/Users/YeungYathin/Desktop/创新实践/复现代码/PSO/M/Q/MQ1',
                     '/Users/YeungYathin/Desktop/创新实践/复现代码/PSO/M/Q/MQ2',
                     '/Users/YeungYathin/Desktop/创新实践/复现代码/PSO/M/Q/MQ3',
                     '/Users/YeungYathin/Desktop/创新实践/复现代码/PSO/M/Q/MQ4',
                     '/Users/YeungYathin/Desktop/创新实践/复现代码/PSO/M/Q/MQ5']

    goal_generation = 250

    for datapath in datapath_list:
        instance = datapath.split('/')[-1].strip()
        output_filename = f'/Users/YeungYathin/Desktop/创新实践/复现代码/PSO/DPSO/OUTPUT_0/{instance}_result.txt'
        f = open(output_filename, 'w+')
        print(f'{instance}')
        print(f'{instance}', file=f)
        
        of = open(f'/Users/YeungYathin/Desktop/创新实践/复现代码/PSO/M/{instance[1]}/{instance}.opt', 'r')
        optimum_value = of.read().strip().split(' ')[-1]
        
        print(f'{goal_generation} {optimum_value}')
        print(f'{goal_generation} {optimum_value}', file=f)

        for i in range(3):
            print(f'{i+1}')
            print(f'{i+1}', file=f)
            main_instance = Main(goal_generation, datapath=datapath)
            found_gbest_facility_vector, found_gbest_fitness_value = main_instance.DPSO()
            print(f'{found_gbest_fitness_value} {time.time()-Main.start}')
            print(f'{found_gbest_fitness_value} {time.time()-Main.start}', file=f)