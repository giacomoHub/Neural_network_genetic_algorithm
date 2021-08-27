import random
import numpy as np
import math
import matplotlib.pyplot as plt 
import timeit
import copy
from sklearn.neural_network import MLPClassifier
from itertools import chain 

class NeuralNetwork:
    weights = [] #weights[layers][nodes_of_layer][nodes_of_previous_layer+1]
    fitness = 0

    def __init__(self):
        self.weights = [] 
        self.fitness = 0

    def feedForward(self, data):
        """return the output nodes after feed-forwarding the data through the network"""
        previous_layer_output = data
        for x in range(0, len(self.weights)): #loop through all layers
            layerOutput = []
            for i in range(0, len(self.weights[x])): #loop through all nodes of x layer
                tempOutput = 0 
                for j in range(0, len(previous_layer_output)): #loop through all nodes of previous layer
                    tempOutput += self.weights[x][i][j] * previous_layer_output[j]
                tempOutput += self.weights[x][i][len(self.weights[x][i])-1] #add the bias
                tempOutput = self.sigmoid(tempOutput) #pass the activation function
                layerOutput.append(tempOutput) #add result to hiddenNodes
            previous_layer_output = layerOutput
        return layerOutput

    def sigmoid(self, x):
        return (1 / (1 + math.exp(-x)))

class PerfomanceStats:
    avg_fit = []
    best_fit_train = []
    best_fit_test = []
    best_fit_error = []
    generation_run_time = []
    gen_index = []

    def __init__(self):
        self.avg_fit = []               #average fitness on the training set
        self.best_fit_train = []        #best fitness on the training data
        self.best_fit_test = []         #best fitness on the test data
        self.generation_run_time = []   #run time for each generation
        self.gen_index = 0              #index of current generation
    
    def get_best_individual(self, population):
        """helper function that returns the best individual in the population"""
        min_fitness = 5000000
        best_individual = NeuralNetwork()
        for member in population:
            if member.fitness < min_fitness:
                min_fitness = member.fitness
                best_individual = copy.deepcopy(member)
        return best_individual

    def get_average_fitness(self, population):
        """helper function that returns the average fitness of a generation"""
        count = 0
        for member in population:
            count += member.fitness
        return round(count/len(population),1)

    def update_stats(self, population, train_data_size, test_data_size, start_time, test_data, test_labels):
        self.gen_index+=1
        #update stats regarding generation time
        end_time = timeit.default_timer()
        self.generation_run_time.append(end_time - start_time)

        current_best_individual = self.get_best_individual(population) #best neural network for this generation
        #update stats regarding training data
        self.avg_fit.append(self.get_average_fitness(population)/train_data_size*100)
        self.best_fit_train.append(current_best_individual.fitness/train_data_size*100)
        #update stats regarding test data
        evaluate_fitness(current_best_individual, test_data, test_labels)
        self.best_fit_test.append(current_best_individual.fitness/test_data_size*100)
        self.best_fit_error.append(current_best_individual.fitness)
        print(f"Generation: {self.gen_index}, Fitness_train: {self.best_fit_train[-1]}, Fitness_test: {self.best_fit_test[-1]}")
         
    def plot_progress(self):
        OUTPUTFILEPATH = "NEX"
        x = [i for i in range(0,self.gen_index)]
        # plotting the points  
        plt.plot(x, self.avg_fit, label='average error') 
        plt.plot(x, self.best_fit_train, label='best individual train data')
        plt.plot(x, self.best_fit_test, label='best individual test data')
        #plt.plot(x, self.best_fit_error, label='error')
        best = min(self.best_fit_test)
        index_of_best = self.best_fit_test.index(best)
        plt.scatter(index_of_best, best, s=50, color='red') #plot the best individual
        plt.legend(loc="upper right")

        # naming the x axis 
        plt.xlabel('Number of Generations') 
        # naming the y axis 
        plt.ylabel('percentage error') 
        # giving a title to my graph 
        plt.title(OUTPUTFILEPATH) 
        # function to show the plot 
        #plt.show() 
        plt.draw()
        plt.pause(0.001)
        #save plot to png
        plotpath = OUTPUTFILEPATH + ".png"
        plt.savefig(plotpath)

    def get_best_result(self):
        return min(self.best_fit_test)

def read_data(file_path, features_count):
    """read data from file and divide into features and labels
    returns the two lists
    """
    feature_columns = [i for i in range(0, features_count)]
    features = np.loadtxt(file_path, delimiter=" ", usecols=feature_columns)
    labels = np.loadtxt(file_path, delimiter=" ", usecols=[features_count])

    return features, labels
    
def split_train_test(data_features, data_labels, train_percentage):
    """shuffle the data and spit it into training and testing data"""
    #shuffle the data
    mappedData = list(zip(data_features, data_labels))
    random.shuffle(mappedData)
    data_features, data_labels = zip(*mappedData)

    #divide data into training and testing sets
    limit = int(round((len(data_features)*train_percentage)*0.01, 1))
    print("70 percent of " + str(len(data_features)) + " is: " + str(limit))
    training_data_features = data_features[:limit]
    training_data_labels = data_labels[:limit]
    test_data_features = data_features[limit:]
    test_data_labels = data_labels[limit:]

    return training_data_features, test_data_features, training_data_labels, test_data_labels

def initialize_population(population_count, min_value, max_value, neural_net_shape):
    """Make a population of population_count lenght and initialize with random numbers
    between min_value and max_value
    """
    pop = []
    for y in range(0, population_count):
        individual = NeuralNetwork()

        for i in range(1, len(neural_net_shape)): #loop through all layers
            layer_weights = []
            for j in range(0, neural_net_shape[i]): #loop through all nodes of one layer
                temp=[] #weights of one node
                for x in range(0, neural_net_shape[i-1]+1): #loop through all nodes of previous layer
                    temp.append(random.uniform(min_value, max_value)) #init single weight
                layer_weights.append(temp)
            individual.weights.append(layer_weights)

        pop.append(individual)
    return pop
    
def normalize_gene(gene_value, minimum_value, maximum_value):
    """Helper function, gets the value of a gene checks if it is in the correct range,
    if not, normalizes the value so it is in range.
    """
    if gene_value > maximum_value:
        gene_value = maximum_value
    elif gene_value < minimum_value:
        gene_value = minimum_value
    return gene_value

def mutate(individual, minimum_value, maximum_value, mutation_rate, mutation_step):
    """randomly mutates the values of one individual"""
    for x in range(0,len(individual.weights)):
        for i in range(0, len(individual.weights[x])):
            for j in range(0, len(individual.weights[x][i])):
                if(random.randint(0,100)<mutation_rate):
                    alter = random.uniform(0.1,mutation_step)
                    add_or_remove = random.randrange(0,2) #0 or 1
                    if(add_or_remove == 0): #decide if to add or remove from the weight
                        individual.weights[x][i][j] += alter
                    else: 
                        individual.weights[x][i][j] -= alter
                    individual.weights[x][i][j] = normalize_gene(individual.weights[x][i][j], minimum_value, maximum_value)
    return individual

def crossover(population, population_count, neural_net_shape):
    """Choses a random crossover point and combines the genes of two parents to make two children"""
    new_population = []
    #for each 2 individuals
    for y in range(0, population_count, 2):
        parents = []
        #flatten the weights of the individuals
        parent1 = list(chain.from_iterable(list(chain.from_iterable(population[y].weights))))
        parent2 = list(chain.from_iterable(list(chain.from_iterable(population[y+1].weights))))
        #find crossover point
        cr_point = random.randint(0,len(parent1)) #crossover point
        #crossover
        temp = parent1[cr_point:]
        parent1[cr_point:] = parent2[cr_point:]
        parent2[cr_point:] = temp
        parents.append(parent1)
        parents.append(parent2)
        #reconstruct the neural nets (unflatten the weights)
        for parent in parents:
            g = 0 #index of the gene
            child = NeuralNetwork()
            for i in range(1, len(neural_net_shape)): #loop through all layers
                layer_weights = []
                for j in range(0, neural_net_shape[i]): #loop through all nodes of one layer
                    temp=[] #weights of one node
                    for x in range(0, neural_net_shape[i-1]+1): #loop through all nodes of previous layer
                        temp.append(parent[g]) #copy single weight
                        g+=1
                    layer_weights.append(temp)
                child.weights.append(layer_weights)
            new_population.append(child)

    #return population
    return new_population

def tournament_selection(population, population_count):
    """Returns a new population of individuals selecting the best ones at random"""
    new_pop = []
    for i in range(0, population_count):
        parent1 = population[random.randint(0,population_count-1)]
        parent2 = population[random.randint(0,population_count-1)]
        if(parent1.fitness < parent2.fitness):
            new_pop.append(copy.deepcopy(parent1))
        else:
            new_pop.append(copy.deepcopy(parent2))
    return new_pop

def roulette_wheel_selection(population, population_count):
    """Returns a new population of individuals selectiong the best ones
    by probability. Fitness is proportional to the probability of being selected"""
    new_pop = []
    #get the sum of all the fitness values
    errorSum = sum([population[i].fitness for i in range(0, population_count)])
    #make new list with all the inverse proportions(sum/fitnessValue[i])
    fitnessList = [errorSum/population[i].fitness for i in range(0, population_count)]
    #normalize values
    inverseErrorSum = sum(fitnessList)
    normalizedFitnessList = [fitnessList[i]/inverseErrorSum for i in range(0, population_count)]

    #select parents
    for i in range(0, population_count):
        #generate random number between 0 and 1
        objective = random.random()
        candidateSum = 0
        j=0
        #sum fitness until objective is reached
        while(candidateSum<objective):
            candidateSum += normalizedFitnessList[j]
            j+=1
        new_pop.append(copy.deepcopy(population[j-1]))

    return new_pop


def evaluate_fitness(individual, data, data_labels):
    """runs neural network against all of the data and sets the error"""
    error = 0

    #loop through each element in data (trainingdata/testdata)
    for i in range(0, len(data)):

        #Run neural net with one instance of the data
        individual_output = individual.feedForward(data[i])

        #check if the result of the neural network calculation is correct
        if(individual_output[0] < 0.5 and data_labels[i]==1.0):
            error+=1
        if(individual_output[0] >= 0.5 and data_labels[i]==0.0):
            error+=1
    individual.fitness = error

def get_best_individual(population):
    """helper function that returns the best individual in the population"""
    min_fitness = 5000000
    best_individual = NeuralNetwork()
    for member in population:
        if member.fitness < min_fitness:
            min_fitness = member.fitness
            best_individual = copy.deepcopy(member)
    return best_individual

def evolve(train_x, test_x, train_y, test_y, FEATURE_COUNT):
    """Evolves a neural network, tuning the the weights"""
    shape = [FEATURE_COUNT,8,6,1] #shape of the neural network
    population = []
    POPULATION_SIZE = 100
    MIN_VALUE = -1
    MAX_VALUE = 1
    MUT_RATE = 4
    MUT_STEP = 0.5
    NUM_OF_GENERATIONS = 100
    previousFitness = 0

    init_mut_rate = MUT_RATE
    init_mut_step = MUT_STEP



    #data for evaluating performance
    performance_data = PerfomanceStats()

    #initialize the population
    print("Initializing population ...")
    population = initialize_population(POPULATION_SIZE, MIN_VALUE, MAX_VALUE, shape)

    #evaluate fitness
    for i in range(0,POPULATION_SIZE):
        evaluate_fitness(population[i], train_x, train_y)

    #evolve
    for i in range(0, NUM_OF_GENERATIONS):
        start = timeit.default_timer()
        best_individual = get_best_individual(population)

        #select
        population = copy.deepcopy(tournament_selection(population, POPULATION_SIZE))
        #population = copy.deepcopy(roulette_wheel_selection(population, POPULATION_SIZE))

        #crossover
        #population = copy.deepcopy(crossover(population,POPULATION_SIZE, shape))

        #mutate
        for member in population:
            member = mutate(member,MIN_VALUE,MAX_VALUE,MUT_RATE,MUT_STEP)

        #carry best individual from previous generation
        population[0] = best_individual

        #check if the algorithm is stalling
        if(population[0].fitness==previousFitness):
            stallingIndex +=1
            if(stallingIndex >=5 and MUT_RATE < 9):
                print("Stalling")
                if(MUT_RATE < 9):
                    MUT_RATE += 1
                    print(f"Mut_rate increased to {MUT_RATE}")
                if(MUT_STEP < 0.6):
                    MUT_STEP += 0.01
        else:
            MUT_RATE = init_mut_rate
            MUT_STEP = init_mut_step
            stallingIndex = 0
        previousFitness = population[0].fitness

        #evaluate
        for i in range(0,POPULATION_SIZE):
            evaluate_fitness(population[i], train_x, train_y)
        
        #update the stats of average fitness and best fitness
        performance_data.update_stats(population,len(train_y),len(test_y),start,test_x, test_y)

        if(i%50==0):
            MUT_STEP -=0.1
            init_mut_step -=0.1

        

    performance_data.plot_progress()
    return performance_data.get_best_result()

def plot_bar(ga_error, bp_error):
    results = []
    results.append(ga_error)
    results.append(bp_error)

    x = np.arange(len(results))

    fig, ax = plt.subplots()
    plt.bar(x, results)
    plt.xticks(x, ('Genetic Algorithm', 'Backpropagation'))

    # naming the y axis 
    plt.ylabel('error') 
    plt.title("Genetic Algorithm vs Backpropagation") 
    plt.draw()
    plt.pause(0.001)
    #save plot to png
    plotpath = "GA_vs_BP2" + ".png"
    plt.savefig(plotpath)


if __name__ == "__main__":

    FILE_PATH = "../Data/data2.txt"
    FEATURE_COUNT = 6
    TRAIN_PERCENTAGE = 70       #percentage of data that will be allocated for training
    ga_best = 0                 #best error percentage for the ga

    #read data
    print("Reading data ...")
    features, labels = read_data(FILE_PATH, FEATURE_COUNT)

    #split data
    print("Splitting data ...")
    train_x, test_x, train_y, test_y = split_train_test(features,labels,TRAIN_PERCENTAGE)

    #run backpropagation training
    clf = MLPClassifier(solver='lbfgs',max_iter=4000, alpha=1e-5,hidden_layer_sizes=(7, 5, 2), random_state=1)
    clf.fit(train_x, train_y)
    error = 0
    results = clf.predict(test_x)
    for i in range(0,len(test_x)):
        if results[i] != test_y[i]:
            error +=1
    percentage_error = (error/len(test_y)*100)
    print("The error for the backpropagation classifier is: " + str(error))
    print("The error percentage is: " + str(percentage_error))

    #run genetic algorithm
    ga_best = evolve(train_x, test_x, train_y, test_y, FEATURE_COUNT)
    print("Neural evolution finished")

    

    #show ga vs bp perfomance
    plot_bar(ga_best, percentage_error)

    #terminate execution
    print("Program execution finished")




