import numpy as np
import errno, sys
from scipy.stats import rankdata
from numba import jit

def weighter (data,i,column,modus):
    """Weight function that assigns a weight to each rank. 

    :param data: Ranking data
    :param i: Rank index
    :param column: Column index
    :param modus: Weighing that is applied, should be on of {"top bottom", "top", "bottom", "middle", "top bottom exp"} 
    :return: Weight corresponding to the rank at index i
    """
    length = len(data)
    x = data[i,column]/length
    if x == 1:
        return None
    if modus == 'top bottom':
        if x<=0.5:
            return 1-2*x
        elif x<1 and x>0.5:
            return 2*x-1
    elif modus == 'top':
            if x<1:
                    return 1-x
    elif modus == 'bottom':
                if x<1:
                    return x        
    elif modus == 'middle':
            if x<=0.5:
                    return 2*x  
            elif x<1 and x >0.5:
                    return 2-2*x
    elif modus == 'top bottom exp':
            if x<1:
                return 4*(x-0.5)**2
    else:
        print('No mode for weighting has been set!')
     
def R(data,x,y,column,d):
    """returns value of the fuzzy order relation R as defined in [1]

    [1] Henzgen, Sascha; Hüllermeier, Eyke  (2015): Weighted Rank Correlation: A Flexible Approach Based on Fuzzy Order Relations.
    In: Machine Learning and Knowledge Discovery in Databases. European Conference, ECML PKDD 2015, Porto, Portugal

    :param data: Ranking data
    :param x: Rank x
    :param y: Rank y
    :param column: column index
    :param d: Distance function
    :return: Value of strict fuzzy ordering 
    """
    if data[x,column]>=data[y,column]:
        return 0
    else:
        return d(data,x,y,column)
    
@jit(nopython=True)
def d_max(data,i,j,column):
    """Pseudo-metric according to eq (11) in the paper

    :param data: Rank data
    :param i: Rank index i 
    :param j: Rank index j
    :param column: column index
    :return: Value of pseudo-metric as a global distance funcion
    """
    return max(data[int(min(data[i,column],data[j,column]))-1:int(max(data[i,column],data[j,column])-1),column+2])

@jit(nopython=True)
def d_sum(data,i,j,column):
    """Pseudo-metric according to eq (12) in the paper

    :param data: Rank data
    :param i: Rank index i 
    :param j: Rank index j
    :param column: Column index
    :return: Value of pseudo-metric as a global distance funcion on 
    """
    return min(1,sum(data[int(min(data[i,column],data[j,column]))-1:int(max(data[i,column],data[j,column])-1),column+2]))
  
def distance_selector(distance):
    """Chooses the distance function for rank correlation coefficient computation

    :param distance: Mode for distance function, either 'max' or 'sum'
    :return: _description_
    """
    if distance=='max':
        return d_max
    elif distance=='sum':
        return d_sum
    else:
        print('Mode for distance calculation is not set!')
        sys.exit(errno.EACCES)

def t_norm_product(a,b):
    """Product t-Norm

    :param a: 
    :param b: 
    :return: T(a,b) = a*b
    """
    return a*b

def t_conorm_product(a,b):
    """Product t-Conorm

    :param a: 
    :param b: 
    :return: ⊥(a,b) = a+b-a*b
    """
    return a+b-a*b

def t_norm_luka(a,b):
    """Łukasiewicz t-Conorm

    :param a: 
    :param b: 
    :return: T(a,b) = max(a+b-1,0)
    """
    return max(a+b-1,0)

def t_conorm_luka(a,b):
    """Łukasiewicz t-Conorm

    :param a: 
    :param b: 
    :return: ⊥(a,b) = min(a+b,1)
    """
    return min(a+b,1)

def tnorm_selector(mode):
    """Selects t-norm for rank correlation coefficient computation

    :param mode: The t-norm to be used, either 'product' or 'luca'
    :return: Tuple of t-norm and corresponding t-conorm
    """
    if mode == 'product':
        tnorm=t_norm_product
        tconorm=t_conorm_product
    elif mode == 'luka':
        tnorm=t_norm_luka
        tconorm=t_conorm_luka
    else:
        print('This mode is not defined')
        sys.exit(errno.EACCES)
    return tnorm,tconorm

# Default parameters
default_configuration=dict(
    weighting='uniform',
    tnorm='product',
    distance='max',
    weights=[]
)

def data_prep(x,y,weights,weighting):
    """Prepares data for rank correlation computation

    :param x: _description_
    :param y: _description_
    :param weights: _description_
    :param weighting: _description_
    :return: _description_
    """
    lenght=len(x)
    
    data1 = np.column_stack((rankdata(x), rankdata(y)))
    sort_indices = np.argsort(data1[:, 0])
    data1=data1[sort_indices]

    data2 = np.column_stack((rankdata(x), rankdata(y)))
    sort_indices2 = np.argsort(data2[:, 1])
    data2=data2[sort_indices2]
    # Auswahl zwischen eigener Gewichtung, oder einer vordefinierten Gewichtung
    if not len(weights):
        weight1=np.ones(lenght)
        weight2=np.ones(lenght)
        if weighting != "uniform":
            for i in range(lenght):   
                weight1[i]=weighter(data1,i,0,weighting)  
                weight2[i]=weighter(data2,i,1,weighting)
        return np.column_stack((data1,weight1,weight2))
    elif len(weights)==(lenght-1):
        weights.append(np.nan)
        return np.column_stack((data1,weights,weights))
    elif len(weights) and len(weights)!=(lenght-1):
        print('Length of weight vector is not n-1!')
        sys.exit(errno.EACCES)   
   
    
def scaled_gamma(data1,data2,**kwargs):
    """Calculates the scaled gamma rank correlation coefficient as proposed by Henzgen and Hüllereier [1]

    [1] Henzgen, Sascha; Hüllermeier, Eyke  (2015): Weighted Rank Correlation: A Flexible Approach Based on Fuzzy Order Relations.
    In: Machine Learning and Knowledge Discovery in Databases. European Conference, ECML PKDD 2015, Porto, Portugal


    :param data1: Rank data a
    :param data2: Rank data b
    :return: Scaled gamma
    """
    kwargs = {**default_configuration, **kwargs}
    weighting=kwargs['weighting']
    weights=kwargs['weights']
    tnorm=kwargs['tnorm']
    distance=kwargs['distance']
    # Data preprocessing
    data =data_prep(data1,data2,weights,weighting)
    # Selection of t-norms and t-conorms
    t_norm,t_conorm=tnorm_selector(tnorm)
    # Selection of distance measure
    d=distance_selector(distance)
    # Initial values 
    concordant = 0
    discordant = 0
    ties = 0
   
    
    # Computation of concordance, discordance and ties for each pair
    for i in range (len(data)):
        for z in range ((i+1),len(data)):
            # Account for ties
            if ((data[i,0] == data[z,0]) or (data[i,1] == data[z,1])):
                ties += 1
            else:
                ties += t_conorm(1-d(data,i,z,0),1-d(data,i,z,1))
                concordant += t_norm(R(data,i,z,0,d),R(data,i,z,1,d)) + t_norm(R(data,z,i,0,d),R(data,z,i,1,d))
                discordant += t_norm(R(data,i,z,0,d),R(data,z,i,1,d)) + t_norm(R(data,z,i,0,d),R(data,i,z,1,d))
    if (concordant == 0) and (discordant == 0):
        return np.nan
    else:
        return (concordant-discordant)/(concordant+discordant)

