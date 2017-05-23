from scipy.sparse import lil_matrix
from numpy.linalg import norm
import numpy as np
from multiprocessing import Pool
from multiprocessing import Process
from scipy import linalg
import time



def xy2i(ix,iy,Nx):
    ii = (iy)*Nx+ix
    return ii

def init_delta(Nx,Ny,delta):
    vec_delta = lil_matrix((Nx*Ny,Nx*Ny))
    for ii in range(Nx*Ny):
        vec_delta[ii,ii] = delta
    return vec_delta


def calc_A(Nx,Ny,mu,vec_delta,aa):
    Ln = Nx*Ny*2
    A = lil_matrix((Ln,Ln))
#    A.setdiag(-mu)
    for ix in range(Nx):        
        for iy in range(Ny):
            ii = xy2i(ix,iy,Nx)
            jx = ix
            jy = iy
            jj = xy2i(jx,jy,Nx)
            A[ii,jj] = -mu
#            print(ii)

# +1 in x direction
            jx = ix +1
            if jx == Nx:
                jx = 0            
            jy = iy
            jj = xy2i(jx,jy,Nx)
            A[ii,jj] = -1.0
            
# -1 in x direction
            jx = ix -1
            if jx == -1:
                jx = Nx-1           
            jy = iy
            jj = xy2i(jx,jy,Nx)
            A[ii,jj] = -1.0

# + 1 in y direction
            jx = ix 
            jy = iy +1
            if jy == Ny:
                jy = 0         
            jj = xy2i(jx,jy,Nx)
            A[ii,jj] = -1.0   

# -1 in y direction
            jx = ix 
            jy = iy -1
            if jy == -1:
                jy = Ny-1
            jj = xy2i(jx,jy,Nx)
            A[ii,jj] = -1.0
    for ii in range(Nx*Ny):
#        print(ii)
        for jj in range(Nx*Ny):
#            print("jj",jj)
            A[ii+Nx*Ny,jj+Nx*Ny] = -A[ii,jj]
            A[ii,jj+Nx*Ny] = vec_delta[ii,jj]
            A[ii+Nx*Ny,jj] = vec_delta[jj,ii]
    
    A = A/aa

#    print(A)
    A = A.tocsr()
#    print(A)
    return A

def calc_A2(A2,Nx,Ny,mu,vec_delta,aa):
    Ln = Nx*Ny*2
    A = A2.tolil()
    A = A*aa
    for ii in range(Nx*Ny):
#        print(ii)
        for jj in range(Nx*Ny):
#            print("jj",jj)
            A[ii,jj+Nx*Ny] = vec_delta[ii,jj]
            A[ii+Nx*Ny,jj] = vec_delta[jj,ii]
    
    A = A/aa

#    print(A)
    A = A.tocsr()
#    print(A)
    return A


def iteration_RSCG(nc,Nx,Ny,U,mu,full,T,omegamax):
    Ln = Nx*Ny*2
    vec_delta = init_delta(Nx,Ny,0.1)
    vec_delta_old = vec_delta

    aa = 1.0
    A = calc_A(Nx,Ny,mu,vec_delta,aa)

    itemax = 100

    for ite in range(itemax):
        if full == False:
            eps = 1e-8
            vec_delta = calc_meanfields_RSCG(eps,A,Nx,Ny,Ln,T,omegamax)
        else:
            vec_delta = calc_meanfields_full_finite(nc,A,Nx,Ny,Ln,T,omegamax)    
            
        vec_delta = vec_delta*U

#        A = calc_A(Nx,Ny,1e-12,vec_delta,10.0)
        A = calc_A2(A,Nx,Ny,1e-12,vec_delta,aa)
        eps = 0.0
        nor = 0.0

        for i in range(Nx*Ny):
            eps += abs(vec_delta[i,i] -vec_delta_old[i,i])**2
            nor += abs(vec_delta_old[i,i])**2
        eps = eps/nor
        print "ite = ",ite,eps
        if eps <= 1e-6:
            print "End",vec_delta[Nx/2,Ny/2]
            break

        vec_delta_old = vec_delta

    return vec_delta
#        print(vec_delta)

def iteration(nc,Nx,Ny,aa,bb,omegac,U,full,mu):
    Ln = Nx*Ny*2
    vec_delta = init_delta(Nx,Ny,0.1)
    vec_delta_old = vec_delta

    A = calc_A(Nx,Ny,mu,vec_delta,10.0)
    itemax = 100

    for ite in range(itemax):
        if full == False:
            vec_delta = calc_meanfields(nc,A,Nx,Ny,Ln,aa,bb,omegac)    
        else:
            vec_delta = calc_meanfields_full(nc,A,Nx,Ny,Ln,aa,bb,omegac)    

        vec_delta = vec_delta*U

#        A = calc_A(Nx,Ny,1e-12,vec_delta,10.0)
        A = calc_A2(A,Nx,Ny,1e-12,vec_delta,10.0)
        eps = 0.0
        nor = 0.0

        for i in range(Nx*Ny):
            eps += abs(vec_delta[i,i] -vec_delta_old[i,i])**2
            nor += abs(vec_delta_old[i,i])**2
        eps = eps/nor
        print "ite = ",ite,eps
        if eps <= 1e-6:
            print "End",vec_delta[Nx/2,Ny/2]
            break

        vec_delta_old = vec_delta

    return vec_delta
#        print(vec_delta)


def RSCG(eps,n_omega,left_i,right_j,vec_sigma,A,Ln):
    
    ci = 1j
    
    vec_x = np.zeros(Ln)
    vec_b =   np.zeros(Ln)
    vec_b[right_j] = 1.0
    vec_r = np.zeros(Ln)
    vec_p = np.zeros(Ln)
    vec_r[right_j] = 1.0
    vec_p[right_j] = 1.0
    vec_Ap = np.zeros(Ln)
    Sigma = vec_b[left_i]
    vec_g = np.zeros(n_omega, dtype=np.complex)
    vec_rhok = np.ones(n_omega, dtype=np.complex)
    vec_rhokp = np.ones(n_omega, dtype=np.complex)    
    vec_rhokm = np.ones(n_omega, dtype=np.complex)
    vec_alpha = np.zeros(n_omega, dtype=np.complex)
    vec_beta = np.zeros(n_omega, dtype=np.complex)
    flag = True
    hi = 1.0
    alpham = 1.0
    betam = 0.0
    vec_Theta = np.zeros(n_omega, dtype=np.complex)
    vec_Pi = np.ones(n_omega, dtype=np.complex)*Sigma

#    a = -A.todense()    
#    x = np.linalg.solve(a, vec_b)
    ep = 1e-15

    while hi > eps:
        vec_Ap = -A.dot(vec_p)
#        print np.dot(vec_p,vec_Ap)
        rsum = np.dot(vec_r,vec_r)
        alpha = rsum/np.dot(vec_p,vec_Ap)
#        print alpha,rsum
        vec_x += alpha*vec_p
        vec_r += - alpha*vec_Ap
        beta = np.dot(vec_r,vec_r)/rsum
        vec_p = vec_r + beta*vec_p
        Sigma = vec_r[left_i]


#        index = vec_rhok > ep
        vec_rhokp = np.where(vec_rhok > ep,vec_rhok*vec_rhokm*alpham/(vec_rhokm*alpham*(1.0+alpha*vec_sigma)+alpha*betam*(vec_rhokm-vec_rhok)),vec_rhok)
        vec_alpha = np.where(vec_rhok> ep, alpha*vec_rhokp/vec_rhok,0.0)
        vec_Theta = vec_Theta+vec_alpha*vec_Pi
        vec_beta = np.where(vec_rhok > ep,((vec_rhokp/vec_rhok)**2)*beta,1.0)
        vec_Pi = np.where(vec_rhok > ep,vec_rhokp*Sigma+ vec_beta*vec_Pi,vec_Pi)

        vec_g = vec_Theta
        vec_rhokm = vec_rhok
        vec_rhok = vec_rhokp

        alpham = alpha
        betam = beta
        hi = rsum

        continue
    


    return vec_g

        
def calc_meanfields_RSCG(eps,A,Nx,Ny,Ln,T,omegamax):
    A = A*1.0
    ci = 1j
    pi = np.arctan(1.0)*4
    vec_delta = lil_matrix((Nx*Ny,Nx*Ny))
#    omegamax = omegac #pi*T(2*n+1), omegac/(T*pi)
    n_omega = (int(omegamax/(T*pi)))/2-1
    
    vec_sigma = np.zeros(2*n_omega, dtype=np.complex)
    
    for n in range(2*n_omega):
        vec_sigma[n] = pi*T*(2.0*(n-n_omega)+1)*ci

    
    for ix in range(Nx):
        for iy in range(Ny):
            ii = xy2i(ix,iy,Nx)
            jj = ii + Nx*Ny
            right_j = jj
            left_i = ii
            vec_g = RSCG(eps,2*n_omega,left_i,right_j,vec_sigma,A,Ln)
            vec_delta[ii,ii] = np.real(T*np.sum(vec_g))

    return vec_delta


def calc_meanfields(nc,A,Nx,Ny,Ln,aa,bb,omegac):    
    vec_delta = lil_matrix((Nx*Ny,Nx*Ny))
    for ix in range(Nx):
        for iy in range(Ny):
            ii = xy2i(ix,iy,Nx)
            jj = ii + Nx*Ny
            right_j = jj
            left_i = ii
            vec_ai = calc_polynomials(nc,left_i,right_j,A,Ln)
            density = calc_meanfield(vec_ai,aa,bb,omegac,nc)
            vec_delta[ii,ii] = density
    return vec_delta


def calc_meanfields_full(nc,A,Nx,Ny,Ln,aa,bb,omegac):    
    a = A.todense()*aa
    w, v = linalg.eigh(a, lower=True, eigvals_only=False, overwrite_a=False, eigvals=None, check_finite=True)    
    vec_delta = lil_matrix((Nx*Ny,Nx*Ny))
    for ix in range(Nx):
        for iy in range(Ny):
            ii = xy2i(ix,iy,Nx)
            jj = ii + Nx*Ny
            delta = 0.0

            for i in range(Nx*Ny*2):
                if w[i] <= 0.0:
                    if abs(w[i]) <= omegac:
                        delta += v[ii,i]*v[jj,i]
            vec_delta[ii,ii] = delta
#            print delta

    return vec_delta

def calc_meanfields_full_finite(nc,A,Nx,Ny,Ln,T,omegamax):    
    a = A.todense()
    w, v = linalg.eigh(a, lower=True, eigvals_only=False, overwrite_a=False, eigvals=None, check_finite=True)    
    vec_delta = lil_matrix((Nx*Ny,Nx*Ny))
    for ix in range(Nx):
        for iy in range(Ny):
            ii = xy2i(ix,iy,Nx)
            jj = ii + Nx*Ny
            delta = 0.0

            delta = calc_green(ii,jj,Nx,Ny,w,v,omegamax,T)
            vec_delta[ii,ii] = delta
#            print delta

    return vec_delta

def calc_green(ii,jj,Nx,Ny,w,v,omegamax,T):
    pi = np.arctan(1.0)*4.0
    ci = 1j
    n_omega = (int(omegamax/(T*pi)))/2-1
    vec_sigma = np.zeros(2*n_omega, dtype=np.complex)
    vec_g = np.zeros(2*n_omega, dtype=np.complex)    
    
    for n in range(2*n_omega):
        vec_sigma[n] = pi*T*(2.0*(n-n_omega)+1)*ci

    for i in range(Nx*Ny*2):
        vec_g += v[ii,i]*v[jj,i]/(vec_sigma - w[i])
    delta = T*np.sum(np.real(vec_g))

    return delta


def calc_meanfield(vec_ai,aa,bb,omegac,nc):    
    ba = np.arccos(-bb/aa)
    omeb = np.arccos(-(omegac+bb)/aa)
    pi = np.arctan(1.0)*4
    density = 0.0
    for j in range(nc-1):
        i = j + 1
        density +=  vec_ai[i]*(np.sin(i*omeb)-np.sin(i*ba))/i
    density += vec_ai[0]*(omeb-ba)/2.0

    density = density*2/pi

    return density



def calc_polynomials(nc,left_i,right_j,A,Ln):
    vec_jnmm = np.zeros(Ln)
    vec_jnm = np.zeros(Ln)
    vec_jn = np.zeros(Ln)
    vec_jn[right_j] = 1.0
    vec_ai = np.zeros(nc)

    for nn in range(nc):
        if nn == 0:
            vec_jnmm = 0.0
            vec_jnm = 0.0
            vec_jn[right_j] = 1.0
        elif nn == 1:
            vec_jn = A.dot(vec_jn)
        else:
            vec_jn = 2.0*A.dot(vec_jnm) - vec_jnmm
        vec_ai[nn] = vec_jn[left_i]
        vec_jnmm = vec_jnm
        vec_jnm = vec_jn
    return vec_ai



def main():
#    for i in range(1,6):
#        print("test")

    nc = 1000
    nx = 10
    ny = 10
    vec_delta = init_delta(nx,ny,0.1)
    mu = -1.0
    A = calc_A(nx,ny,mu,vec_delta,10.0)
#    print A
#    quit()


    U = -4.0
    mu = -1.5



    a = A.todense()
    w, v = linalg.eigh(a, lower=True, eigvals_only=False, overwrite_a=False, eigvals=None, check_finite=True)
#    print(w 

    vec_ai = calc_polynomials(nc,1,1+nx*ny,A,nx*ny*2)
#    print(vec_ai)
    density = calc_meanfield(vec_ai,10.0,0.0,10.0,nc)
#    print(density)

    aa = 10.0
    bb = 0.0
    omegac = 10.0
    Ln = nx*ny*2
    
 #   vec_delta = calc_meanfields(nc,A,nx,ny,Ln,aa,bb,omegac)    
#    print(vec_delta)



    print "Exact diagonalization"
    start = time.time()
    iteration(nc,nx,ny,aa,bb,omegac,U,True,mu)
    elapsed_time = time.time()-start
    print "elapsed_time",elapsed_time


    print "Chebyshev polynomial method"
    start = time.time()
    iteration(nc,nx,ny,aa,bb,omegac,U,False,mu)
    elapsed_time = time.time()-start
    print "elapsed_time",elapsed_time

    print "RSCG method"
    T = 0.05
    pi = np.arctan(1.0)*4.0    
    omegamax = pi*240
    start = time.time()
    iteration_RSCG(nc,nx,ny,U,mu,False,T,omegamax)
    elapsed_time = time.time()-start
    print "elapsed_time",elapsed_time

    print "Green-function-based Exact diagonalization"
    start = time.time()
    iteration_RSCG(nc,nx,ny,U,mu,True,T,omegamax)
    elapsed_time = time.time()-start
    print "elapsed_time",elapsed_time





#    x = np.zeros((nx*ny*2))
#    for i in range(nx*ny*2):
#        x[i] = i + 1
#    print(x)
#    x = matrix([[1],[1],[1],[2],[2],[3]])

#    y = A.dot(x)
#    print(y)

    

if __name__ == "__main__":
    main()
