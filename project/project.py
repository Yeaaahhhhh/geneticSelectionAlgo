import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

# fitness calculation
def calFitness(line,dis_matrix):
    dis_sum = 0
    dis = 0
    for i in range(len(line)):
        if i<len(line)-1:
            dis = dis_matrix.loc[line[i],line[i+1]]#
            dis_sum = dis_sum+dis
        else:
            dis = dis_matrix.loc[line[i],line[0]]
            dis_sum = dis_sum+dis
    return round(dis_sum,1)

# tournament select
def tournament_select(pops,popsize,fits,tournament_size):
    new_pops,new_fits = [],[]
    while len(new_pops)<len(pops):
        tournament_list = random.sample(range(0,popsize),tournament_size)
        tournament_fit = [fits[i] for i in tournament_list]

        tournament_df = pd.DataFrame([tournament_list,tournament_fit]).transpose().sort_values(by=1).reset_index(drop=True)

        fit = tournament_df.iloc[0,1]
        pop = pops[int(tournament_df.iloc[0,0])]
        new_pops.append(pop)
        new_fits.append(fit)
    return new_pops,new_fits


def crossover(popsize,parent1_pops,parent2_pops,pc):
    child_pops = []
    for i in range(popsize):

        child = [None]*len(parent1_pops[i])
        parent1 = parent1_pops[i]
        parent2 = parent2_pops[i]
        if random.random() >= pc:
            child = parent1.copy()
            random.shuffle(child)
        else:
            start_pos = random.randint(0,len(parent1)-1)
            end_pos = random.randint(0,len(parent1)-1)
            if start_pos>end_pos:
                tem_pop = start_pos
                start_pos = end_pos
                end_pos = tem_pop
            child[start_pos:end_pos+1] = parent1[start_pos:end_pos+1].copy()
            # parent2 -> child
            list1 = list(range(end_pos+1,len(parent2)))
            list2 = list(range(0,start_pos))
            list_index = list1+list2
            j = -1
            for i in list_index:
                for j in range(j+1,len(parent2)):
                    if parent2[j] not in child:
                        child[i] = parent2[j]
                        break
        child_pops.append(child)
    return child_pops


def mutate(pops,pm):
    pops_mutate = []
    for i in range(len(pops)):
        pop = pops[i].copy()
        t = random.randint(1,5)
        count = 0
        while count < t:
            if random.random() < pm: 
                    mut_pos1 = random.randint(0,len(pop)-1)  
                    mut_pos2 = random.randint(0,len(pop)-1)
                    if mut_pos1 != mut_pos2:
                        tem = pop[mut_pos1]
                        pop[mut_pos1] = pop[mut_pos2]
                        pop[mut_pos2] = tem
            pops_mutate.append(pop)
            count +=1
    return pops_mutate


def draw_path(line,CityCoordinates):
    x,y= [],[]
    for i in line:
        Coordinate = CityCoordinates[i]
        x.append(Coordinate[0])
        y.append(Coordinate[1])
    x.append(x[0])
    y.append(y[0])
    plt.plot(x, y,'r-', color='#4169E1', alpha=0.8, linewidth=0.8)
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.savefig('result1.png')
    plt.show()

if __name__ == '__main__':

    CityNum = 20 
    MinCoordinate = 0
    MaxCoordinate = 101
    #GA
    generation = 100 
    popsize = 100   
    tournament_size = 5 
    pc = 0.95   #crossOver proba
    pm = 0.1    #mutation proba

    # set city size, we use a set of numbers for test, while we can generate a set of random city Coordinates.
    # CityCoordinates = [(random.randint(MinCoordinate,MaxCoordinate),random.randint(MinCoordinate,MaxCoordinate)) for i in range(CityNum)]
    CityCoordinates = [(88, 16),(42, 76),(5, 76),(69, 13),(73, 56),(100, 100),(22, 92),(48, 74),(73, 46),(39, 1),(51, 75),(92, 2),(101, 44),(55, 26),(71, 27),(42, 81),(51, 91),(89, 54),(33, 18),(40, 78)]  #test cases
    dis_matrix = pd.DataFrame(data=None,columns=range(len(CityCoordinates)),index=range(len(CityCoordinates)))
    for i in range(len(CityCoordinates)):
        xi,yi = CityCoordinates[i][0],CityCoordinates[i][1]
        for j in range(len(CityCoordinates)):
            xj,yj = CityCoordinates[j][0],CityCoordinates[j][1]
            dis_matrix.iloc[i,j] = round(math.sqrt((xi-xj)**2+(yi-yj)**2),2)  #generate a matrix with coordinate's distance.

    iteration = 0
    pops = [random.sample([i for i in list(range(len(CityCoordinates)))],len(CityCoordinates)) for j in range(popsize)]


    fits = [None]*popsize
    for i in range(popsize):
        fits[i] = calFitness(pops[i],dis_matrix)

    best_fit = min(fits)
    best_pop = pops[fits.index(best_fit)]
    print('first opimal %.1f' % (best_fit))
    best_fit_list = []
    best_fit_list.append(best_fit)

    while iteration <= generation:

        pop1,fits1 = tournament_select(pops,popsize,fits,tournament_size)
        pop2,fits2 = tournament_select(pops,popsize,fits,tournament_size)
        #cross over
        child_pops = crossover(popsize,pop1,pop2,pc)
        #mutation
        child_pops = mutate(child_pops,pm)
        #calculate fitness
        child_fits = [None]*popsize
        for i in range(popsize):
            child_fits[i] = calFitness(child_pops[i],dis_matrix) 
        #keep the optimal
        for i in range(popsize):
            if fits[i] > child_fits[i]:
                fits[i] = child_fits[i]
                pops[i] = child_pops[i]

        if best_fit>min(fits):
            best_fit = min(fits)
            best_pop = pops[fits.index(best_fit)]

        best_fit_list.append(best_fit)

        print('the %d generation optimal %.1f' % (iteration, best_fit))
        iteration += 1

    
    #print(best_pop)
    
    draw_path(best_pop,CityCoordinates)
    
    iters = list(range(len(best_fit_list)))
    plt.plot(iters, best_fit_list, 'r-', color='#4169E1', alpha=0.8, linewidth=0.8)
    plt.xlabel('iteration time')
    plt.savefig('result2.png')
    #plt.show()    