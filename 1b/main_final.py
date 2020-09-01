import numpy as np
from sklearn.linear_model import LassoCV


#DEFINE THE TRANSFORMATION OF X WITH THE GIVEN FEATURES 
def make_x(x):
	#LINEAR AND QUADRATIC
    add_x = np.hstack((x,x**2))
	#ADD EXPONENTIAL
    add_x = np.hstack((add_x, np.exp(x)))
	#ADD COSINUS
    add_x = np.hstack((add_x, np.cos(x)))
	#ADD CONSTANT
    add_x = np.hstack((add_x, np.ones([x.shape[0],1])))
	#RETURN THE NEW X
    return add_x


def main():
    
    #TRAIN DATA
    data = np.genfromtxt('train.csv', delimiter = ',')
    
    #REMOVE INDEX COL
    data = data[1:,1:]
    
    #SEPARATE Y FROM X
    x_init = data[:,1:]
    y_final = data[:,0]
        
    #ADD FEATURES TO X
    x_final = make_x(x_init)

	#DO THE LASSO REGRESSION 
    regression = LassoCV(eps=1e-3, n_alphas=1000, cv=3, tol=0.0001, fit_intercept = False, max_iter = 100000000)
    
	#FIT THE LINEAR MODEL
    regression.fit(x_final,y_final)
 
    #COMPUTE BEST WEIGHTS
    output_weights=regression.coef_
        
    #OUTPUT RESULTS TO THE CSV FILE
    np.savetxt('Final.csv', output_weights)

    return

if __name__ == "__main__":

    main()

    
