import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import tensorflow as tf

class Bayesian_GPLVM_Collapsed(object):


	# Bayesian Gaussian Process Latent Variable Model for Dimensionality Reduction

	# plots high-dimensional data in 2D scatter plot

    def __init__(self, Y, latent_dim,no_of_inducing_points,x_mean=None, kern=None,x_prior_mean=None,x_prior_var=None):

        self.sess = tf.Session()
        self.Y_train = tf.placeholder(tf.float32,shape=(None,Y.shape[1]))
		
        self.Y =Y
        self.latent_dim = 	latent_dim
        self.num_data = Y.shape[0]
        self.no_of_inducing_points = no_of_inducing_points
        assert Y.shape[1] > self.latent_dim

        # variational distributions q(X) to approximate the true posterior distribution P(X|Y)
        niu = []
        s_diag_cov = []
        for iter_temp in range(self.num_data):
            instance_niu = tf.Variable(tf.random_normal(shape=(self.latent_dim,1)),dtype=tf.float32)
            instance_s_diag_cov = tf.Variable(tf.diag(tf.ones(shape=(self.latent_dim,),dtype=tf.float32)),dtype=tf.float32)
            niu.append(instance_niu)
            s_diag_cov.append(instance_s_diag_cov)
        # variational distribution q(U) to approximate the tru posterior distribution P(U|Y)
        u_mean_variational = []
        u_var_variational = []
        for iteratie_latent in range(self.latent_dim):
            instance_u_mean = tf.Variable(tf.random_normal(shape=(self.no_of_inducing_points,1)),dtype=tf.float32)
            instance_u_var = tf.Variable(tf.random_normal(shape=(self.no_of_inducing_points,1)),dtype=tf.float32)
            u_mean_variational.append(instance_u_mean)
            u_var_variational.append(instance_u_var)

        self.Z = tf.Variable(tf.random_normal(shape=(self.no_of_inducing_points,self.latent_dim),dtype=tf.float32))

        if x_prior_mean is None:
            x_prior_mean = tf.zeros(shape=(self.num_data,self.latent_dim),dtype=tf.float32)
		
        self.x_prior_mean = x_prior_mean	
		
        if x_prior_var is None:
            x_prior_var = tf.ones(shape=(self.num_data,self.latent_dim),dtype=tf.float32)

        self.x_prior_var = x_prior_var
        self.lengthscales = tf.Variable(np.ones(shape=(self.latent_dim,)),dtype=tf.float32)
        self.variance = tf.Variable(np.ones(shape=(1,)),dtype=tf.float32)
        self.niu=niu
        self.s_diag_cov = s_diag_cov

        self.u_mean_variational = u_mean_variational
        self.u_var_variational = u_var_variational
    
    def eye(self,N):

        return tf.diag(tf.ones(tf.pack([N,]),dtype=tf.float32))


	# TODO - implement the non-ARD kernel as well
    def RBF(self,X1,X2,ARD=True):
	        
        X1 = X1 / self.lengthscales
        X2 = X2 / self.lengthscales
        X1s = tf.reduce_sum(tf.square(X1),1)
        X2s = tf.reduce_sum(tf.square(X2),1)       

        return self.variance * tf.exp(-(-2.0 * tf.matmul(X1,tf.transpose(X2)) + tf.reshape(X1s,(-1,1)) + tf.reshape(X2s,(1,-1)))/2)        


    def get_psi_statistics(self):

        psi0 = self.num_data * self.variance

    	#psi1 = np.zeros(shape=(self.num_data,self.no_of_inducing_points))

        psi1_lista_num_data = defaultdict()
        psi1_num_data_packed = []

    	# computing psi1
        for iter_n in range(self.num_data):

            psi1_lista_num_data[iter_n] = []

            for iter_m in range(self.no_of_inducing_points):
                temp_product = self.variance

                for iter_latent in range(self.latent_dim):
                    temp_product = temp_product * tf.exp(-0.5 * (self.lengthscales[iter_latent] * tf.pow(self.niu[iter_n][iter_latent]-self.Z[iter_m,iter_latent],2)) / (self.lengthscales[iter_latent] * self.s_diag_cov[iter_n][iter_latent,iter_latent]))
                    temp_product + temp_product * (1 /  tf.sqrt(self.lengthscales[iter_latent]*self.s_diag_cov[iter_n][iter_latent,iter_latent]+1.0))

                psi1_lista_num_data[iter_n].append(temp_product)
            psi1_num_data_packed.append(tf.pack(psi1_lista_num_data[iter_n]))


        psi1 = tf.pack(psi1_num_data_packed,axis=1)

    	# computing psi2

    	#psi2 = np.zeros(shape=(self.no_of_inducing_points,self.no_of_inducing_points))

        psi2_dict = defaultdict()

        for iter_m1 in range(self.no_of_inducing_points):
            psi2_dict[iter_m1] = defaultdict()
            for iter_m2 in range(self.no_of_inducing_points):
                psi2_dict[iter_m1][iter_m2] = tf.constant(value=0.0)

        psi2_fragmentat_dict = []
        for iter_n in range(self.num_data):
            psi2_fragmentat_dict[iter_n] = []
            for iter_m in range(self.no_of_inducing_points):
                psi2_fragmentat_dict[iter_n][iter_m] = []

        for iter_n in range(self.num_data):
    	    for iter_m1 in range(self.no_of_inducing_points):
    		    for iter_m2 in range(self.no_of_inducing_points):

    			    temp_product = tf.pow(self.variance,2)
    	            for iter_latent in range(self.num_latent):
    				    z_tilda = (self.Z[iter_m1,iter_latent] + self.Z[iter_m2,iter_latent])/2.0
    				    temp_product = temp_product * exp(-(self.lengthscales[iter_latent]*tf.pow(self.Z[iter_m1,iter_latent] - self.Z[iter_m2,iter_latent],2)/4.0) - (self.lengthscales[iter_latent] * tf.pow(self.niu[iter_n][iter_latent] - z_tilda,2))/(2*self.lengthscales[iter_latent]*self.s_diag_cov[iter_n][iter_latent,iter_latent]+1))
    				    temp_product = temp_product * (1/ tf.sqrt(2*self.lengthscales[iter_latent]*self.s_diag_cov[iter_n][iter_latent,iter_latent]+1.0))

                    psi2_dict[iter_m1][iter_m2] = tf.add(psi2_dict[iter_m1][iter_m2],temp_product)
                    psi2_fragmentat_dict[iter_n][iter_m1].append(temp_product)

        psi2_fragmentat = []
        for iter_n in range(self.num_data):
            temporar = np.squeeze(psi_fragmentat_dict[0])
            for iter_m in range(1,self.no_of_inducing_points):
                temporar = tf.pack([temporar,np.squeeze(psi2_fragmentat_dict[iter_m])])
            psi2_fragmentat.append(temporar)


        return psi0, psi1,psi2_fragmentat




    def build_likelihood(self):


    	likelihood = 0 
    	Kuu = self.RBF(self.Z,self.Z) + self.eye(self.no_of_inducing_points)

    	psi0, psi1, psi2 = self.get_psi_statistics()

    	for iter_latent_dim in range(self.latent_dim):

    			likelihood += -self.num_data * 0.5 * tf.log(self.variance) + 0.5 * tf.log(tf.matrix_determinant(Kuu))
    			lieklihood +=  - self.num_data * 0.5 * tf.log(2.0 * np.pi) - 0.5 * tf.log(tf.matrix_determinant((1/self.variance)*psi2 + Kuu))

    			W = (1/self.variance) * np.identity(self.num_data) - tf.pow(1/self.variance,2) * tf.matmul(tf.matmul(psi1,tf.matrix_inverse(tf.add((1/self.variance)*psi2,Kuu))),tf.transpose(psi1)) 
    			likelihood += -0.5 * tf.matmul(tf.matmul(tf.transpose(self.Y[:,iter_latent_dim]),W),self.Y[:,iter_latent_dim])

    			likelihood += -0.5*(1/self.variance) *psi0 + 1/(2*self.variance) * tf.trace(tf.matmul(tf.matrix_inverse(Kuu),psi2))


    	# add Kullback-Liebler Divergence term

    	for iter_data in range(self.num_data):
    		likelihood += - 0.5 * tf.trace(tf.add(tf.add(tf.matmul(self.niu[iter_data],tf.transpose(self.niu[iter_data])),self.s_diag_cov[iter_data]),tf.log(self.s_diag_cov))) 

    	likelihood += 0.5 * self.num_data * self.latent_dim


    	return likelihood

    #def build_predict(self):

    def build_likelihood_uncollapsed(self):

        likelihood = 0
        Kuu = self.RBF(self.Z,self.Z) + self.eye(self.no_of_inducing_points)
        Kuf = self.RBF(self.Z,self.x_prior_mean)


        L = tf.cholesky(Kuu)
        L_inverse = tf.matrix_inverse(L)
        psi0, psi1, psi2 = self.get_psi_statistics()
        Kuu_inv = tf.matrix_inverse(Kuu)

        for iter_i in range(self.num_data):
            for iter_j in range(self.latent_dim):



                likelihood += -0.5 * tf.log(2 * np.pi*self.variance)
                temp1 = tf.matrix_triangular_solve(L,psi1[iter_i,:],lower=True)
                temp2 = tf.matrix_triangular_solve(L,tf.matmul(Kuu,self.u_mean_variational[iter_j]),lower=True)
                likelihood += - (0.5 / self.variance) * tf.matmul(temp1,temp2)
                temp3 = tf.matrix_triangular_solve(L,self.u_mean_variational[iter_j],lower=True)
                temp4 = tf.matrix_triangular_solve(L,tf.matmul(Kuu,psi2[iter_i,:]),lower=True)

                likelihood += - (0.5 / self.variance ) * tf.trace(tf.matmul(tf.matmul(tf.matmul(tf.transpose(L_inverse),temp2),tf.transpose(temp3)),temp4))
                #temp5 = tf.matrix_triangular_solve(L,tf.matrix_diag(self.u_var_variational),lower=True)
                likelihood += - (0.5 / self.variance) * tf.trace(tf.matmul(tf.matmul(tf.matmul(Kuu_inv,self.u_var_variational[iter_j]),Kuu_inv),psi2[iter_i]))


                likelihood += - (0.5 / self.variance) * ((psi0 / self.num_data) - tf.trace(tf.matmul(psi2[iter_i],Kuu_inv)))


        # add Kullback-Liebler divergence term from KL(q(X)||P(X))

        for iter_data in range(self.num_data):
            likelihood += - 0.5 * tf.trace(tf.add(tf.add(tf.matmul(self.niu[iter_data],tf.transpose(self.niu[iter_data])),self.s_diag_cov[iter_data]),tf.log(self.s_diag_cov))) 

        likelihood += 0.5 * self.num_data * self.latent_dim

        # add KUllback-LIebler divergence term from KL(q(U)||p(U))

        likelihood += - 0.5 * self.latent_dim * tf.reduce_sum(tf.log(tf.diag_part(Kuu)))

        for iter_q in range(self.latent_dim):

            likelihood += - 0.5 * tf.trace(tf.matmul(Kuu_inv,self.u_var_variational[iter_q])+tf.matmul(tf.matmul(tf.transpose(self.u_mean_variational[iter_q]),Kuu_inv),self.u_mean_variational[iter_q]))

            likelihood += 0.5 * tf.reduce_sum(tf.log(self.u_var_variational[iter_q]))

        likelihood += self.num_data * self.latent_dim * 0.5



    def session_TF(self,ytrain,no_of_iterations = 100):

        init_op = tf.initialize_all_variables()

        self.sess.run(init_op)

    	cost = self.build_likelihood_uncollapsed()
    	train_op = tf.train.AdagradOptimizer(0.0001).minimize(cost)
        print ' am terminat de compilat'

    	for iter_training in range(no_of_iterations):

    		self.sess.run(train_op,feed_dict ={self.Y_train : ytrain})
    		current_cost_value = self.sess.run(cost,feed_dict={self.Y_train:ytrain})
    		print('iteration '+str(iter_training)+' : '+str(current_cost_value))


    	highest_indices = self.lengthscales.argsort()[-2:][::-1]

    	plt.figure()
    	dimension1 = []
    	dimension2 = []
    	for iter_data in range(self.num_data):
    		dimension1.append(self.niu[iter_data][highest_indices[0]])
    		dimension2.append(self.niu[iter_data][highest_indices[1]])

    	plt.figure()
    	plt.plot(dimension1,dimension2)
    	plt.show()



if __name__ == '__main__':

	# main function 


    # load data
    y_training = np.loadtxt('DataTrn.txt')

    print 'am terminat de load data'
    GPLVM_object = Bayesian_GPLVM_Collapsed(Y =y_training , latent_dim  =2 ,no_of_inducing_points = 100)
    GPLVM_object.session_TF(ytrain=y_training)











    







