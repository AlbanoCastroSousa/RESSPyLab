# -*- coding: utf-8 -*-

"""Main module."""

import numpy as np
import pandas as pd
import codecs
import numdifftools.nd_algopy as nda


def polyfit_with_fixed_points(n, x, y, xf, yf) :
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x**np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf**np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:n + 1]


def getTestData(fileName,A):
    
    f=codecs.open(fileName, 'rb',encoding='cp1252 ')
    localMeasurements=f.readlines()
    f.close

    localMeasurements=localMeasurements[7:]

    aux=[]

    for i in localMeasurements:
        aux2=[]
        for j in i[:-2].split('\t'):
            #if j=='':
            #    continue
            #aux2.append(float(j.replace(' ','')))
            aux2.append(j)
        aux.append(aux2)

    header=aux[0]
    header[-1]='Target_2_Y'
    header[-2]='Target_2_X'
    header[-3]='Target_1_Y'
    header[-4]='Target_1_X'

    aux=aux[1:]

    testData=pd.DataFrame(aux,columns=header)

    testData=testData.rename(columns={'ai0':'F_kN'})

    #A=np.pi*(9.97**2+9.98**2*2+9.99**2)/4./4.

    testData['F_kN']=testData['F_kN'].astype('float64') 

    testData['Target_1_X']=testData['Target_1_X'].astype('float64') 
    testData['Target_1_Y']=testData['Target_1_Y'].astype('float64') 
    testData['Target_2_X']=testData['Target_2_X'].astype('float64') 
    testData['Target_2_Y']=testData['Target_2_Y'].astype('float64') 

    testData['Sigma_eng']=testData['F_kN']/A*1000

    L0=testData['Target_2_Y'][0]-testData['Target_1_Y'][0]

    testData['e_eng']=(testData['Target_2_Y']-testData['Target_1_Y']-L0)/L0

    testData['e_true']=np.log(1.+testData['e_eng'])

    testData['Sigma_true']=testData['Sigma_eng']*(1.+testData['e_eng'])
    
    return testData


def cleanTestData(testData,e_indexes,window):

    xload=[0.]
    yload=[0.]
  

    for i in np.arange(len(e_indexes)-1):


        start=e_indexes[i]
        finish=e_indexes[i+1]

        for j in np.arange((np.floor((finish-start)/(window*1.)))+1):

            index1=int(start+window*j)
            if j == (np.floor((finish-start)/(window*1.))):
                index2=finish
            else:
                index2=int(start+window*(j+1)) # beginning and end values of the indices to interpolate

            x=testData['e_true'][index1:index2].values
            y=testData['Sigma_true'][index1:index2].values

            xf=[xload[-1]]#[testData['e_true'][index1],testData['e_true'][index2]]#fixed points in x
            yf=[yload[-1]]#[testData['Sigma_true'][index1],testData['Sigma_true'][index2]] #fixed points in y

            params = polyfit_with_fixed_points(2, x , y, xf, yf) # name of the function says it all
            poly = np.polynomial.Polynomial(params) # function with the parameters that were estimated

            xx=np.linspace(xload[-1],testData['e_true'][index2],3) # resolution in x#np.linspace(testData['e_true'][index1],testData['e_true'][index2],3) # resolution in x


            #ax.plot(xx,poly(xx))
            for i in range(len(xx)):
                xload.append(xx[i])
                yload.append(poly(xx)[i])

    testClean=pd.DataFrame(np.array([xload,yload]).transpose(),columns=['e_true','Sigma_true'])
    testClean['delta_e']=testClean['e_true'].shift(-1)-testClean['e_true']

    testClean['e_cumsum']=np.cumsum(np.abs(testClean['e_true'].shift(-1)-testClean['e_true']))
    
    return testClean



def errorTest_scl(x_sol,testClean):

    sy_0=x_sol[1]*1.0
    E=x_sol[0]*1.0
    Q=x_sol[2]*1.0
    b=x_sol[3]*1.0

    nBack=int((len(x_sol)-4)/2)

    chabCoef=[]

    for i in range(nBack):
        chabCoef.append([x_sol[4+(2*i)]*1.0,x_sol[5+(2*i)]*1.0])

    sigma=0.0
    ep_eq=0.0
    e_p=0.0
    e_el=0.0
    e=0.0

    alphaN=np.zeros(nBack, dtype = object)*1.0
    sy=sy_0*1.0

    Tol=1e-10

    e_true_calc=[0]
    sigma_calc=[0]
    ep_eq_calc=[0]


    lineCounter=0

    Phi_test=0.0
    
    sum_abs_de=0.0

    sigma_SimN=0.0
    sigma_testN=0.0

    loading=testClean['delta_e'].dropna().values

    for de in loading:

        lineCounter=lineCounter+1

        e=e+de
        #e_el=e_el+de

        alpha=np.sum(alphaN)

        sigma=sigma+E*de

        phi=(sigma-alpha)**2-sy**2

        ep_eq_n_1=ep_eq*1.0

        alphaN_1=alphaN*1.0

        sy_N_1=sy*1.0

        sigma_SimN_1=sigma_SimN*1.0
        sigma_testN_1=sigma_testN*1.0


        if phi > Tol:

            nitMax=1000

            #print 'Initiation:',phi
            #print 'e,de,ep:','%e'%e,'%e'%de,'%e'%e_p
            #print 'sigma,s-a,sy',sigma,sigma-alpha,sy

            for nit in np.arange(nitMax):

                #print nit,phi,dep,sigma,sy,alpha

                aux=E*1.0
                for k in np.arange(nBack):
                    aux=aux+np.sign(sigma-alpha)*chabCoef[k][0]-chabCoef[k][1]*alphaN[k]

                dit=(-2.*alpha+2.*sigma)*aux+2.*sy*Q*b*np.exp(-b*ep_eq)

                dep=(phi/(dit))

                scale=1.0

                # scale to ensure that the Newton step does not overshoot

                if abs(dep)>abs(sigma/E):
                    dep=np.sign(dep)*0.95*abs(sigma/E)
                    scale=0.95*abs(sigma/E)/abs(dep)


                


                ## Update variables ##

                ep_eq=ep_eq+np.abs(dep)

                e_p=e_p+dep

                sigma=sigma-E*dep

                sy=sy_0+Q*(1.-np.exp(-b*ep_eq))

                for k in np.arange(nBack):
                    c_k=chabCoef[k][0]
                    gam_k=chabCoef[k][1]
                    alphaN[k]=np.sign(sigma-alpha)*c_k/gam_k-(np.sign(sigma-alpha)*c_k/gam_k-alphaN_1[k])*np.exp(-gam_k*(ep_eq-ep_eq_n_1))

                alpha=np.sum(alphaN)

                phi=(sigma-alpha)**2-sy**2
                #print nit,phi#,dep,ep_eq,sigma,sy,alpha,alphaN,alphaN_1

                if abs(phi) <Tol:
                    #print 'Final:',phi,nit
                    #print 'e,de:','%e'%e,'%e'%de,'%e'%e_p
                    #print 'sigma,s-a,sy',sigma,sigma-alpha,sy
                    #print ' '
                    break

                if nit-2==nitMax:
                    print ('Warning convergence not reached in nonlinear loop!!!')


        sigma_SimN=sigma*1.0
        sigma_testN=testClean['Sigma_true'].iloc[lineCounter]

        e_true_calc.append(e)
        sigma_calc.append(sigma)
        ep_eq_calc.append(ep_eq)

        #difSigN=sigma-testClean
        
        sum_abs_de=sum_abs_de+np.abs(de)

        # Square of the area under the increment with the trapezoidal rule

        Phi_test=Phi_test+np.abs(de)*((sigma_SimN-sigma_testN)**2+(sigma_SimN_1-sigma_testN_1)**2)/2.

    Phi_test=Phi_test/sum_abs_de
    
    return Phi_test



def errorEnsemble_nda(x_sol):
    
    Phi_ensemble=0.0
    
    for cleanedTest in arrayCleanTests:
        
        Phi_ensemble=Phi_ensemble+errorTest_scl(x_sol,cleanedTest)
    
    #barrier for function for zero constraints
    
    for barrier in range(len(x_sol)-2):
        
        Phi_ensemble=Phi_ensemble+1/x_sol[-(barrier+1)]**2
        
    return Phi_ensemble


##### Steihaug-Toint truncated conjugated gradient method #####

def steihaug(Q,b,Delta):
    
    
    x_1=np.zeros(len(b))
    d=-b*1.0
    
    flag=0
    
    for i in range(len(b)+1):
        
        x_prev=x_1
        
        if np.dot(d,np.dot(Q,d))<0:
            
            flag=1
            
            a_=np.dot(d,d)
            b_=np.dot(2*x_1,d)
            c_=np.dot(x_1,x_1)-Delta**2
            
            lambda_=(-b_+np.sqrt(b_**2-4*a_*c_))/(2*a_)
            
            x_stei=x_1+lambda_*d
            
            break
        
        
        alpha=-np.dot(d,np.dot(Q,x_1)+b)/np.dot(d,np.dot(Q,d))
        
        x_1=x_prev+alpha*d
        
        if np.sqrt(np.dot(x_1,x_1))>Delta:
            
            flag=1
            
            a_=np.dot(d,d)
            b_=np.dot(2*x_prev,d)
            c_=np.dot(x_prev,x_prev)-Delta**2
            
            lambda_=(-b_+np.sqrt(b_**2-4*a_*c_))/(2*a_)
            
            x_stei=x_prev+lambda_*d
            
            break
         

        
        beta=np.dot(np.dot(Q,x_1)+b,np.dot(Q,x_1)+b)/np.dot(np.dot(Q,x_1)+b,np.dot(Q,x_1)+b)
        
        d=-np.dot(Q,x_1)-b+beta*d
            
    if flag==0:
        x_stei=x_1
    
    return x_stei



def NTR_SVD_Solver(f,df,Hf,x):

    ##### Newton's method with trust region with Steihaug-Toint truncated conjugated gradient method with SVD preconditioning ######

    # Initialization

    dk=np.zeros(len(x))

    Delta=10
    Tol=1e-10

    eta1=0.01
    eta2=0.9

    nitNMTRmax=int(1e6)

    gradPhi_fun=df
    HessPhi_fun=Hf

    gradPhi_test=gradPhi_fun(x)
    HessPhi_test=HessPhi_fun(x)


    for nit in range(nitNMTRmax):


        Phi_test_k=f(x)*1.0
        
        # Solve trust-region sub-problem with steihaug-toint method and a preconditioned hessian
        
        U,D,V=np.linalg.svd(HessPhi_test)
    
        #Filtering out ill-conditioned space
        
        D_S=[]
        D_S_inv=[]
        
        for sing in D:
            if sing <1e-15*D.max():
                D_S.append(0.)
                D_S_inv.append(0)
            else:
                D_S.append(sing**-0.5)
                D_S_inv.append(sing**0.5)

                
        D_S=np.array(D_S)
        D_S_inv=np.array(D_S_inv)
            
        
        S=np.dot(np.dot(V.transpose(),np.diag(D_S)),U.transpose())
        S_inv=np.dot(np.dot(U,np.diag(D_S_inv)),V)
        
        HessPhi_test_S=np.dot(np.dot(S.transpose(),HessPhi_test),S)
        
        gradPhi_test_S=np.dot(S.transpose(),gradPhi_test)

        dk=steihaug(HessPhi_test_S,gradPhi_test_S,Delta)
        
        # Bring back the scaled step
        
        dk=np.dot(S,dk)
        
        # Prevent the step from over-shooting the barrier
        
        x_trial=x+dk
        
        barrier_scale=1.0
            
        if np.min(x_trial)<0:
            ind=np.argmin(x_trial)
            barrier_scale=(1e-2-x[ind])/dk[ind]

        dk=barrier_scale*dk
        
        
        
        model_k=Phi_test_k*1.0
        
        model_k1=Phi_test_k+np.dot(dk,gradPhi_test)+np.dot(dk,np.dot(HessPhi_test,dk))
        
        Phi_test_k1=f(x+dk)*1.0
        
        rho=(Phi_test_k-Phi_test_k1)/(model_k-model_k1)
        
        if ((model_k-model_k1)<1e-14 and (Phi_test_k-Phi_test_k1)>0) or np.abs((Phi_test_k-Phi_test_k1)/Phi_test_k)<1e-14:

            x=x+dk
                    
            gradPhi_test=gradPhi_fun(x)
            HessPhi_test=HessPhi_fun(x)
            
            
            if rho>=0.9:
                Delta=2.*Delta
            

        elif rho<eta1 or (Phi_test_k-Phi_test_k1)<0:

            Delta=0.5*np.sqrt(np.dot(np.dot(S_inv,dk),np.dot(S_inv,dk)))
            
        else:

            x=x+dk
                    
            gradPhi_test=gradPhi_fun(x)
            HessPhi_test=HessPhi_fun(x)
            
            
            if rho>=0.9:
                Delta=2.*Delta
                
        norm_grad=np.sqrt(np.dot(gradPhi_test,gradPhi_test))

        
        print (nit,Phi_test_k,norm_grad,rho,Delta,(model_k-model_k1))
        
        if norm_grad < Tol:
            break
        
        

    return x


def VCopt(x_0,listCleanTests):

    global arrayCleanTests

    arrayCleanTests=list(listCleanTests)

    grad_error_fun=nda.Gradient(errorEnsemble_nda)

    Hess_error_fun=nda.Hessian(errorEnsemble_nda)

    x_sol=NTR_SVD_Solver(errorEnsemble_nda,grad_error_fun,Hess_error_fun,x_0)

    return x_sol
