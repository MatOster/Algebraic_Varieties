import numpy as np
from sympy.tensor.array import derive_by_array


precision = 10**(-20) # set numerical precision
epsilon_eig_val = 10**(-17) # set cut of eigenvalues
espilon_lin_approx = 10**(-15) # set epsilon for error of tangent cone
delta = 10**(-5) #set delta corresponding to epsilon of tangent cone

#theoretically for given epsilon there is a delta, such that each point in a delta neighborhood around x_0 has error less than epsilon to tangent cone


def df(x):
    df = [[2*x[0],2*x[1],0,0],[2*(x[0]-x[2]),2*(x[1]-x[3]),-2*(x[0]-x[2]),-2*(x[1]-x[3])],[0,0,2*(x[2]-1),2*x[3]]]#,[2*(x[0]-1),2*x[1],0,0]]
    return df

def f(x):
    f= [x[0]**2+x[1]**2-1,(x[0]-x[2])**2+(x[1]-x[3])**2-1,(x[2]-1)**2+x[3]**2-1]#, (x[0]-1)**2+(x[1])**2-2]
    return f



# use extended Newton-Method to find zero's of polynomial system by randomly pertubating already known zero
def newton(x0):
    x=x0
    i=0

    maxIt=1000
    pert1 =delta* np.random.rand(len(x), 1)     # set delta neighborhood to already known zero
    x=x+.1*pert1[0]
    while(i<maxIt and np.linalg.norm(f(x))>precision):
        psi=df(x)
        inv=np.linalg.pinv(psi)                 # use Monroe-Pseudoinverse for Newton-Method to be applicalbe in an underdetermined system
        y=np.dot(inv,f(x))
        x=np.add(x,-y)
        i+=1

    print(i)
    return x

#create a point cloud of samples of the algebraic variety locally
def sampling(number_of_samples,x):
    y = []
    for i in range(number_of_samples):
        y.append(list(newton(x)))
    return np.array(y)

#calculate mean value of all coordinate directions
def mean_vector(data):
    mean_vector = []
    for i in range(len(data[0])):
        mean_vector.append(np.mean(data[:,i]))
    return mean_vector

#calculate covariance matrix
def scatter_matrix(data):
    scatter  = np.zeros((len(data[0]),len(data[0])))
    mean = mean_vector(data)
    for i in range(len(data)):
        scatter += np.dot(data[i,:]-mean, (data[i,:]-mean).T)
    return scatter

#calcualte Eigenvalue of covariance matrix
def eigenvalue(scat_matrix):
    return np.linalg.eig(scat_matrix)

#calculate dimension of algebraic variety by finding the nummber of non-zero Eigenvalues within some precision
def dimensionality_check(data):

    scat_matrix=scatter_matrix(data)
    eig_val=eigenvalue(scat_matrix)[0]
    if all(eig < precision for eig in eig_val):
        print("The system has dimension zero or is singular")
        #Assumption: singularities always have tangent cone (translated affine cone (i.e. commplex case in Cox Little O'Shea))
        # and hence have 0 covariance eigenvalues

        if(is_Isolated(data,delta)): print('The system has dimension zero')
        else: print('The system is singular')
    else:
        poss_dim =sum(eig_val>precision)
        print("the system has positive dimension smaller than "+str(poss_dim))
        #Assupmtion: Since all singularites are cone like all cases of positive covariance will be at a regular point

        # calculate_dimension(scat_matrix,poss_dim)

# # core of dimensionality_check(): Find true dimension by checking error of sample points to linear subspaces induced by PCA
# def calculate_dimension(data,poss_dim):
#     subspace=eigenvalue()
#     #check if a priori known solution is singularity
#     for n in range(poss_dim):
#         if (is_onSubspace(subspace)[0]):
#             return is_onSubspace[1]
#         else:
#             #subspace=
#             calculate_dimension(data,poss_dim)
#
# def is_onSubspace():
#


# check if  zero is of zero dimesnion by finding neighborhood such thhat there is only one zero
def is_Isolated(data,delta):

    if(data ): return True
    else: return False









x=[0,1,1,1]
print(f(x))
print(df(x))
sample =sampling(10,x)
print(sample)
print(mean_vector(sample))
print(scatter_matrix(sample))
print(eigenvalue(scatter_matrix(sample))[0])
dimensionality_check(sample)


