import numpy as np
from sympy.tensor.array import derive_by_array
import Function_BMN

# Idee:
#theoretically for given epsilon there is a delta, such that each point in a delta neighborhood around x_0 has error less than epsilon to tangent cone
#approximate variety linearly by using PCA (i.e. reduce dimension by directions corresponding to vanishing eigenvalues of scatter matrix of sampling)
#Difficulties: Identify singular points. Solve this by exhaustively check for isolated zeros (in case for intersections of variety)


#Numerical Parameters
precision = 10**(-20) # set numerical precision
epsilon_eig_val = 10**(-17) # set cut of eigenvalues
espilon_lin_approx = 10**(-15) # set epsilon for error of tangent cone

#Structure Parameters
delta = 1 # prescribed periodicity of structure




#Extended Ralphson-Newton method - perturbed starting point
# use extended Newton-Method to find zero's of polynomial system by randomly pertubating already known zero
def newton_pert(x0):
    x=x0
    i=0
    maxIt=1000
    
    
    eta = 10**(-5)                              # radius of pertubation
    bias = np.ones((len(x),1))                  # shift uniformly distribution to intervall -1 to 1
    
    pert1 =2*eta* (np.random.rand(len(x), 1)[0]-.5*bias[0])     
    x=x+pert1
    while(i<maxIt and np.linalg.norm(Function_BMN.f(x))>precision):
        psi=Funciton_BMN.df(x)
        inv=np.linalg.pinv(psi)                 # use Monroe-Pseudoinverse for Newton-Method to be applicalbe in an underdetermined system
        y=np.dot(inv,Function_BMN.f(x))
        x=np.add(x,-y)
        i+=1
    return np.array(x)

# Extended Ralphson-Newton method - deterministic starting point
def newton(x0):
    i = 0
    x = x0

    maxIt = 10000

    while (i < maxIt and np.linalg.norm(Function_BMN.f(x)) > precision):
        psi = Funciton_BMN.df(x)
        inv = np.linalg.pinv(
            psi)  # use Monroe-Pseudoinverse for Newton-Method to be applicable in an underdetermined system
        y = np.array(np.dot(inv, Function_BMN.f(x)))
        x = np.add(x, -y[0])
        i += 1

    return np.array(x)

# gradient descent (already specialized for Lagrangian = squared norm of defining polynomial system)
def grad_des(x):
    i = 0
    max_It = 10000
    x = np.array(x)
    xnow = 0 * x
    gradnow = 0 * x
    norm =1
    while (i < max_It and norm > 5+10**(-15)):
        grad = np.array(Function_BMN.gradient(x))
        norm =np.linalg.norm(list(grad))
        
        #adaptive stepsize after Barzilai and Borwein
        xprev = xnow
        xnow = x
        s = xnow - xprev
        gradprev = gradnow
        gradnow = grad
        g = gradnow - gradprev
        alpha1 = .1 * s.dot(g) / np.linalg.norm(g)**2
        x = x - alpha1 * grad
        i += 1

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
    return 1(len(data)-1)*scatter

#calcualte Eigenvalue of covariance matrix
def eigenvalue(scat_matrix):
    return np.linalg.eig(scat_matrix)

# check if zero is isolated by surounding zero with starting points for Newton, such that every distinct new zero is closer to one of the starting points than the original zero
# cover a simplex plus origin with spheres of radius varying delta on centres of simplex plus origin with known zero centred at
# if there is any other zero in this simplex plus origin at least on vertex of the simplex would converge there under newton

#Attention!! this method only works, if Netwon method finds closest zero to given starting point. This is not always the case
def is_zero_dimensional(x0):
    m = len(x0)
    v = []
    v_centre = []
    eta = 10 ** (-8)  # radius of neighborhood

    # create centre point
    for i in range(m):
        v_centre.append(1 / (m + 1))

    # create points on vertices of tetrahedron
    for i in range(m + 1):
        # create coordinates of higherdim. tetrahedron
        coordinate = []
        for j in range(m):
            k = 0
            if (j + 1 == i): k = 1
            coordinate.append(k)
            # shift centre to x0
        # print(coordinate)
        coordinate = eta * np.array(coordinate) + np.array(x0) + eta * np.array(v_centre)  # translate tetrahedron to right place (ne
        v.append(list(coordinate))

    v.append((v_centre))

    # calculate closest zero to vertices of tetrahedron
    res = []

    for i in range(m + 1):
        res.append(newton((v[i])))

    var = 0
    for i in range(m + 1):
        var += np.linalg.norm(res[i] - x0)

    if (1 / (m + 1) * var < 10**(-15)):
        print(1 / (m + 1) * var)
        return True
    else:
        return False
    
#Alternative method to check for isolation
#Find minimum of the squared norm of the defining polynomial system under the constraint to have cetain distance to known zero. 
#If the minimum is zero close to known zero, it is maybe not isolated
def is_zero_dimensional_grad_des(x0,eta):

    xr = np.random.rand(len(x0),1)[0]
    bias  =1/2*np.ones((len(x0),1))
    xrand = 2*(xr-bias[0])
    x= eta/np.linalg.norm(xrand)*xrand+np.array(x0)
    #gradient descent on Lagrange functional
    y = grad_des(x)
    if (np.linalg.norm(f(y))<10**(-15)):

        return True
    else: return False

#calculate dimension of algebraic variety by finding the nummber of non-zero Eigenvalues within some precision
def dimensionality_check(x0, sample_number):
    if(is_zero_dimensional(x0)):
        print('The system has dimension zero')
        return 0
    else:
        data = sampling(sample_number,x0)
        scat_matrix=scatter_matrix(data)
        eig_val=eigenvalue(scat_matrix)[0]
        print(eig_val)
        if all(eig < precision for eig in eig_val):
            print("The system is singular")
            return -1
            #Assumption: singularities always have tangent cone (translated affine cone (i.e. commplex case in Cox Little O'Shea))
            # and hence have 0 covariance eigenvalues

            #NEW! assumption might not be true, in the sense that there are singularieties with non-zero covariance.

            # if(is_Isolated(data,delta)): print('The system has dimension zero')
            # else: print('The system is singular')
        else:
            poss_dim =sum(eig_val>precision)
            print("the system has positive dimension smaller than "+str(poss_dim))
            return poss_dim
            #Assupmtion: Since all singularites are cone like all cases of positive covariance will be at a regular point

            # calculate_dimension(scat_matrix,poss_dim)

            #NEW! Slice system such that directions with zero variance can be checked on singularities

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


# intersect variety along eigenvectors of scatter matrix corresponding to vanishing eigenvalues
def is_Singular(data,delta):

    if(data ): return True
    else: return False

x= [
    0.75, 0.65, 0.0,
    0.6, 0.25,
    0.75, 0.35, 0.5,
    0.75, 0.9, 0.5,
    0.75, 0.1, 0.0,
    0.15, 0.5, 0.25,
    0.5, 0.75, 0.9,
    0.0, 0.75, 0.1,
    0.5, 0.75, 0.35,
    0.5, 0.25, 0.6,
    0.5, 0.25, 0.15,
    0.4, 0.0, 0.25,
    0.25, 0.15, 0.5,
    0.25, 0.85, 0.0,
    0.1, 0.0, 0.75,
    0.25, 0.4, 0.0,
    0.85, 0.0, 0.25,
    0.25, 0.6, 0.5,
    0.65, 0.0, 0.75,
    0.0, 0.25, 0.4,
    0.0, 0.25, 0.85,
    0.0, 0.75, 0.65,
    1, 1
]

dimensionality_check(x,100000)













