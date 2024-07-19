from sklearn.gaussian_process.kernels import Matern, Kernel#, Sum, WhiteKernel
import numpy as np
import scipy as sc



# Definition of exponential-decay kernel for local GPs
class ExponentialDecayNoiseKernel(Kernel):
    def __init__(self, alpha=1.0,beta=0.5,noise=0.1):
        self.beta = beta
        self.alpha = alpha
        self.noise = noise
    def __call__(self, X, Y=None,eval_gradient=False):
        if Y is None:
            Y = X
        X=np.array(X)
        Y=np.array(Y)
        return (((self.beta**self.alpha)/((X[None,:]+Y[:,None])+self.beta)**self.alpha) + self.noise*np.where(X[None,:]-Y[:,None] == 0, 1, 0)).T
    def diag(self, X):
        return np.diag(self(X))
    def is_stationary(self):
        return False
    
# class WarpedMatern(Kernel):
#     def __init__(self, nu=2.5, a=1.0,b=0.5,theta=0.1):
#         self.beta = beta
#         self.alpha = alpha
#         self.noise = noise
#         self.nu = nu
#     def __call__(self, X, Y=None,eval_gradient=False):
#         if Y is None:
#             Y = X
#         X=np.array(X)
#         Y=np.array(Y)
#         return (((self.beta**self.alpha)/((X[None,:]+Y[:,None])+self.beta)**self.alpha) + self.noise*np.where(X[None,:]-Y[:,None] == 0, 1, 0)).T
#     def diag(self, X):
#         return np.diag(self(X))
#     def is_stationary(self):
#         return False

def compute_ks(x,new_x,kernel):
    k_x = kernel(x)
    k_x_star = kernel(x,new_x)
    k_star_star = kernel(new_x)
    return k_x,k_x_star,k_star_star

def compute_k_star_star(new_x,kernel):
    return kernel(new_x)

def compute_k_t_inv_and_lambda(x,x_dict,kernel_local):
    k_t_invs=[]
    lambdas=[]
    for x_n in x:
        k_t=kernel_local(x_dict['_'.join([str(c) for c in x_n])][0])
        # print(x_n)
        # print(x_dict['_'.join([str(c) for c in x_n])])
        # print(k_t)
        k_t_invs.append(np.linalg.inv(k_t))
        lambdas.append(np.sum(k_t_invs[-1]))
    return sc.linalg.block_diag(*k_t_invs),sc.linalg.block_diag(*lambdas)

def compute_o_osum(config_list,config_dict):
    o=[]
    o_sum=0
    o_sums=[0]
    for config in config_list:
        observations=config_dict['_'.join([str(c) for c in config])][1]
        o.append(np.ones((len(observations),1)))
        o_sum+=len(observations)
        o_sums.append(o_sum)
    return sc.linalg.block_diag(*o),o_sums

def compute_gamma(o,o_sum,k_t_inv,y,m):
    ktinv_yom=k_t_inv@(y-o@m)
    result=np.empty(0)
    for i in range(o.shape[1]):
        result=np.append(result,np.sum(ktinv_yom[o_sum[i]:o_sum[i+1]]))
    return result

def constant_mean(y_vec,x):
    print(np.average(y_vec))
    return np.average(y_vec)*np.ones(x.shape[0])

def compute_y_vector(config_list,config_dict):
    y_vec=np.empty(0)
    for config in config_list:
        observations=config_dict['_'.join([str(c) for c in config])][1]
        y_vec=np.append(y_vec,observations)
    return y_vec

def compute_c(k_x_inv,lambd):
    return np.linalg.inv(k_x_inv+lambd)

def compute_omega(k_t_n_star,k_t_n_inv):
    obs_steps=k_t_n_inv.shape[0]
    predict_steps=k_t_n_star.shape[1]
    return np.ones(predict_steps)-k_t_n_star.T@k_t_n_inv@np.ones(obs_steps)

def compute_mu(m,c,gamma):
    return m+c@gamma

# Equation 14/19:
def compute_mu_x_star(m,k_x_star,k_x_inv,mu,means_vec):
    return k_x_star.T@k_x_inv@(mu-means_vec)+m
def compute_sigma_x_star_star(k_x_star_star,k_x_star,k_x,lambd_inv):
    return k_x_star_star-k_x_star.T@np.linalg.inv(k_x+lambd_inv)@k_x_star

# Equation 16/21:
def compute_mu_n_star_new(mu_n,x_new):
    return mu_n*np.ones(x_new.shape[0])
def compute_sigma_n_star_new(k_t_star_star,sigma_star_star):
    return k_t_star_star+np.identity(k_t_star_star.shape[0])*sigma_star_star

# Equation 15/20:
def compute_mu_n_star_ex(k_t_n_star,k_t_n_inv,y_n,omega_n,mu_n):
    return k_t_n_star.T@k_t_n_inv@y_n+(omega_n*mu_n)
def compute_sigma_n_star_ex(k_t_n_star_star,k_t_n_star,k_t_n_inv,omega_n,c_nn):
    return k_t_n_star_star-k_t_n_star.T@k_t_n_inv@k_t_n_star+omega_n@(c_nn[0]*omega_n.T)

def compute_entropy(mu_vector,var_vector,n_samples):
    var_mat=np.diag(var_vector)
    mc_bins=np.zeros(mu_vector.shape[0])
    for _ in range(n_samples):
        global_gp_samples = np.random.multivariate_normal(mu_vector,var_mat)
        mc_bins[np.argmin(global_gp_samples)]+=1/n_samples
    return sc.stats.entropy(mc_bins)

def compute_ei_at_x(mu,var,best_mu):
    z=(best_mu-mu)/np.sqrt(var)
    return np.sqrt(var)*(z*sc.stats.norm.cdf(z))+sc.stats.norm.pdf(z)

def compute_mu_var_cov_c(observed_configs_list,observed_configs_dicts,new_configs,kernel_global,kernel_local):
    # Calculate the mean and variance at the asymptote for each config using equation 19
    # print("Computing ks")
    k_x,k_x_star,k_x_star_star = compute_ks(observed_configs_list,new_configs,kernel_global)
    # print("Inverting k_x")
    k_x_inv = np.linalg.inv(k_x)
    # print("Computing k_t and inverse and lambda")
    k_t_inv,lambd=compute_k_t_inv_and_lambda(observed_configs_list,observed_configs_dicts,kernel_local)
    # print("Computing y")
    y_vec = compute_y_vector(observed_configs_list,observed_configs_dicts)
    # print("Computing means")
    means_vec = constant_mean(y_vec,observed_configs_list)
    # print(means_vec)
    # print("Computing o")
    o,o_sum=compute_o_osum(observed_configs_list,observed_configs_dicts)
    # print("Inverting lambda")
    lambd_inv = np.linalg.inv(lambd)
    # print("Computing c")
    c=compute_c(k_x_inv,lambd)
    # print("Computing gamma")
    gamma = compute_gamma(o,o_sum,k_t_inv,y_vec,means_vec)
    # print("Computing mu_global")
    mu_global = compute_mu(means_vec,c,gamma)
    # print(mu_global)
    # print("Computing mu")
    mu = compute_mu_x_star(constant_mean(y_vec,new_configs),k_x_star,k_x_inv,mu_global,means_vec)
    # print("Computing cov")
    cov = compute_sigma_x_star_star(k_x_star_star,k_x_star,k_x,lambd_inv)
    var=np.diag(cov)
    return mu,var,cov,c


def init_configs(local_function,hp_bounds,n_init_configs=10,n_init_epochs=5):

    bounds=hp_bounds
    # Start by observing n_init_configs random configurations for n_init_epochs epoch each
    observed_configs_dicts={}
    observed_configs_list=np.empty((0,len(bounds.keys())))
    for _ in range(n_init_configs):
        while True:
            new_config=np.empty(0)
            for key,_ in bounds.items():
                new_config=np.append(new_config,np.round(np.random.uniform(bounds[key][0],bounds[key][1]),2))
            if not np.any(np.all(np.isin(observed_configs_list,new_config),axis=1)):
                break
        # Observe the new configuration for n_init_epochs epochs
        f_space = np.linspace(0,n_init_epochs-1,n_init_epochs)
        experimental_data=local_function(new_config,f_space)
        observed_configs_dicts['_'.join([str(config) for config in new_config])]=(f_space,experimental_data)
        observed_configs_list=np.vstack([new_config,observed_configs_list])

    observed_configs_list=np.array(observed_configs_list)
    # print(self.observed_configs_dicts)
    # print(self.observed_configs_list)
    return observed_configs_list,observed_configs_dicts

def update_configs(local_function,observed_configs_list,observed_configs_dicts,new_config,new_epochs):
    if np.any(np.all(np.isin(observed_configs_list,new_config),axis=1)):
        results=local_function(new_config,new_epochs)
        old_config_entry=observed_configs_dicts['_'.join([str(c) for c in new_config])]
        observed_configs_dicts['_'.join([str(c) for c in new_config])]=(np.append(old_config_entry[0],new_epochs),np.append(old_config_entry[1],results))
        return observed_configs_list,observed_configs_dicts

    results=local_function(new_config,new_epochs)
    observed_configs_list=np.vstack([observed_configs_list,new_config])
    observed_configs_dicts['_'.join([str(c) for c in new_config])]=(new_epochs,results)
    return observed_configs_list,observed_configs_dicts
        


class FreezeThaw():
    def __init__(self,hp_bounds,observed_configs_list,observed_configs_dicts,global_kernel=None,local_kernel=None,alpha=1.0,beta=0.5,noise=0.1):
        self.kernel_global = Matern(nu=2.5) if global_kernel is None else global_kernel
        self.kernel_local = ExponentialDecayNoiseKernel(alpha=alpha,beta=beta,noise=noise) if local_kernel is None else local_kernel
        # self.mean=inferred_mean
        self.bounds=hp_bounds

        self.observed_configs_list=observed_configs_list
        self.observed_configs_dicts=observed_configs_dicts
        # print(self.observed_configs_dicts)
        # print(self.observed_configs_list)


    def iterate(self,b_old=10,b_new=3,n_samples_mc=1000,n_fant=5,ei_n_samples = 1000,pred_epoch = 1):
        # Fill the basket with configs
        basket_new=np.empty((0,len(self.bounds.keys())))
        basket_old=np.empty((0,len(self.bounds.keys())))
        basket_new_mu_var=[]
        basket_old_mu_var=[]
        basket_old_c=[]

        # Get the best yet observed configuration
        best_observation = np.min(np.concatenate([self.observed_configs_dicts['_'.join([str(c) for c in config])][1] for config in self.observed_configs_list]))
        # print(f"Best observation: {best_observation}")

        # Calculate EI for many configs to find the best ones
        # Sample N_EI_SAMPLES new configurations
        ei_configs = []
        for _ in range(ei_n_samples):
            print(f"Sampling new config {len(ei_configs)+1}/{ei_n_samples}",end='\r',flush=True)
            while True:
                new_config=np.empty(0)
                for key,_ in self.bounds.items():
                    new_config=np.append(new_config,np.round(np.random.uniform(self.bounds[key][0],self.bounds[key][1]),2))
                if not np.any(np.all(np.isin(self.observed_configs_list,new_config),axis=1)):
                    break
            ei_configs.append(new_config)
        if len(basket_old)<b_old:
            ei_configs = np.concatenate([ei_configs,self.observed_configs_list])
        ei_configs=np.array(ei_configs)
        # print(ei_configs)
        # print("\nComputing mu, var for all ei-configs")
        mu,var,_,c=compute_mu_var_cov_c(self.observed_configs_list,self.observed_configs_dicts,ei_configs,self.kernel_global,self.kernel_local)
        # print(f"μx*:\n{mu}")
        # print(f"Σx**:\n{var}")

        # Calculate the EI scores for each config using equation 3
        ei_scores=compute_ei_at_x(mu,var,best_observation)
        sort_indices = np.argsort(ei_scores)[::-1]
        ei_configs_ranked = ei_configs[sort_indices]
        # print(ei_configs_ranked)

        # Greedily choose the best HP-config using Equation 19 & 3 until B_OLD existing configs and B_NEW new configs are found
        for n_sample,sampled_ei_config in enumerate(ei_configs_ranked):
            print(f"Adding configs to baskets {basket_new.shape[0]+basket_old.shape[0]}/{b_old+b_new}               ",end='\r',flush=True)
            # Sample another config from EI and try to add it to the basket
            # If it is already in the basket, or the basket is full, skip it
            if not np.any(np.all(np.isin(self.observed_configs_list,sampled_ei_config),axis=1)) and (basket_new.shape[0]==0 or b_new>basket_new.shape[0] and not np.any(np.all(np.isin(basket_new,sampled_ei_config),axis=1))):
                # print(f"Adding new config to basket_new: {sampled_EI_config}")
                basket_new=np.vstack([basket_new,sampled_ei_config])
                basket_new_mu_var.append([mu[n_sample],var[n_sample]])
            elif np.any(np.all(np.isin(self.observed_configs_list,sampled_ei_config),axis=1)) and (basket_old.shape[0]==0 or b_old>basket_old.shape[0] and not np.any(np.all(np.isin(basket_old,sampled_ei_config),axis=1))):
                # print(f"Adding new config to basket_old: {sampled_EI_config}")
                basket_old=np.vstack([basket_old,sampled_ei_config])
                basket_old_c.append(c[np.where((self.observed_configs_list==sampled_ei_config).all(axis=1)),np.where((self.observed_configs_list==sampled_ei_config).all(axis=1))])
                basket_old_mu_var.append([mu[n_sample],var[n_sample]])

        basket_new_mu_var=np.array(basket_new_mu_var)
        basket_old_mu_var=np.array(basket_old_mu_var)
        basket_old_c=np.array(basket_old_c)
        baskets_combined=np.vstack([basket_new,basket_old])


        # print(f"New basket:\n{basket_new}")
        # print(f"Old basket:\n{basket_old}")
        # print(f"New basket mu var:\n{basket_new_mu_var}")
        # print(f"Old basket mu var:\n{basket_old_mu_var}")

        # Compute the entropy of the basket via Monte Carlo sampling
        baskets_mu_var=np.concatenate([basket_new_mu_var,basket_old_mu_var])
        h_p_min=compute_entropy(baskets_mu_var[:,0],baskets_mu_var[:,1],n_samples_mc)
        # print(f"Entropy of P_min: {h_p_min}")
        a=np.zeros(basket_new.shape[0]+basket_old.shape[0])
        pred_epochs=np.linspace(0,pred_epoch-1,pred_epoch)
        epochs_list=[]

        # For each config in the basket, N_FANT times fantasize an observation and recompute the information gain from it, collecting it in a
        for k_config,chosen_config in enumerate(basket_new):
            # Fantasize an observation using Equation 21
            # print(f"Chosen new config: {chosen_config}")
            # print(f"Fantasizing new config: {chosen_config} ({k_config}/{basket_new.shape[0]})                                ",end='\r',flush=True)
            mu_n_star_new = compute_mu_n_star_new(basket_new_mu_var[k_config][0],pred_epochs)
            # print(f"μn* (new):\n{mu_n_star_new}")
            sigma_n_star_new = compute_sigma_n_star_new(compute_k_star_star(pred_epochs,self.kernel_local),basket_new_mu_var[k_config][1])
            # print(f"Σn* (new):\n{sigma_n_star_new}")
            epochs_list.append(pred_epochs)

            for _ in range(n_fant):
                # Fantasize an observation using the mu and sigma of the new config
                fantasized_observation = np.random.multivariate_normal(mu_n_star_new,np.sqrt(sigma_n_star_new))

                # Compute the global mus and sigmas now including the fantasized observation
                observations_incl_list=np.vstack([self.observed_configs_list,chosen_config])
                observations_incl_dicts=self.observed_configs_dicts.copy()
                observations_incl_dicts['_'.join([str(c) for c in chosen_config])]=(pred_epochs,fantasized_observation)

                mu_y,var_y,_,_=compute_mu_var_cov_c(observations_incl_list,observations_incl_dicts,baskets_combined,self.kernel_global,self.kernel_local)
                # print(f"μx*:\n{mu_y}")
                # print(f"Σx**:\n{var_y}")

                # Compute the new entropy of p_min_y, H(P_min_y)
                h_p_min_y=compute_entropy(mu_y,var_y,n_samples_mc)
                # print(f"Entropy of P_min: {h_p_min_y}")
                a[k_config]+=(h_p_min_y-h_p_min)/n_fant

        for k_config,chosen_config in enumerate(basket_old):
            # print(f"Chosen existing config: {chosen_config}")
            # print(f"Fantasizing existing config: {chosen_config} ({k_config}/{basket_old.shape[0]})                                ",end='\r',flush=True)
            pred_epochs_n=pred_epochs+1+self.observed_configs_dicts['_'.join([str(c) for c in chosen_config])][0][-1]
            # print(pred_epochs_n)
            epochs_list.append(pred_epochs_n)
            curve_n=self.observed_configs_dicts['_'.join([str(c) for c in chosen_config])]

            k_t_n,k_t_n_star,k_t_n_star_star=compute_ks(curve_n[0],pred_epochs_n,self.kernel_local)
            k_t_n_inv=np.linalg.inv(k_t_n)
            omega_n = compute_omega(k_t_n_star,k_t_n_inv)
            mu_n_star_ex = compute_mu_n_star_ex(k_t_n_star,k_t_n_inv,curve_n[1],omega_n,basket_old_mu_var[k_config][0])
            # print(f"μn* (existing):\n{mu_n_star_ex}")
            sigma_n_star_ex = compute_sigma_n_star_ex(k_t_n_star_star,k_t_n_star,k_t_n_inv,omega_n,basket_old_c[k_config])
            # print(f"Σn* (existing):\n{sigma_n_star_ex}")

            for _ in range(n_fant):
                # Fantasize an observation using Equation 20
                fantasized_observation = np.random.multivariate_normal(mu_n_star_ex,np.sqrt(sigma_n_star_ex))
                # print(f"Fantasized observation: {fantasized_observation}")

                # Compute the global mus and sigmas now including the fantasized observation
                observations_incl_list=self.observed_configs_list
                observations_incl_dicts=self.observed_configs_dicts.copy()
                old_config_entry=observations_incl_dicts['_'.join([str(c) for c in chosen_config])]
                observations_incl_dicts['_'.join([str(c) for c in chosen_config])]=(np.append(old_config_entry[0],pred_epochs_n),np.append(old_config_entry[1],fantasized_observation))
                # print(observations_incl_dicts['_'.join([str(c) for c in chosen_config])])

                mu_y,var_y,_,_=compute_mu_var_cov_c(observations_incl_list,observations_incl_dicts,baskets_combined,self.kernel_global,self.kernel_local)
                # print(f"μx*:\n{mu}")
                # print(f"Σx**:\n{var}")

                # Compute the new entropy of p_min_y, H(P_min_y)
                h_p_min_y=compute_entropy(mu_y,var_y,n_samples_mc)
                # print(f"Entropy of P_min: {h_p_min_y}")

                a[k_config+b_new]+=(h_p_min_y-h_p_min)/n_fant


        # Select the config with the highest information gain
        best_config=baskets_combined[np.argmax(a)]
        print(f"Next config: {best_config} ({'new' if np.argmax(a)<b_new else 'old'})")
        return best_config,epochs_list[np.argmax(a)]

    def predict_global(self,configs):
        configs=np.array(configs)
        mu,var,_,_=compute_mu_var_cov_c(self.observed_configs_list,self.observed_configs_dicts,configs,self.kernel_global,self.kernel_local)
        return mu,var
    
    def predict_local(self,config,epochs):
        mu,_,_,c=compute_mu_var_cov_c(self.observed_configs_list,self.observed_configs_dicts,np.array([config]),self.kernel_global,self.kernel_local)
        curve_n=self.observed_configs_dicts['_'.join([str(c) for c in config])]
        k_t_n,k_t_n_star,k_t_n_star_star=compute_ks(curve_n[0],epochs,self.kernel_local)
        k_t_n_inv=np.linalg.inv(k_t_n)
        omega_n = compute_omega(k_t_n_star,k_t_n_inv)
        mu_n_star_ex = compute_mu_n_star_ex(k_t_n_star,k_t_n_inv,curve_n[1],omega_n,mu)
        sigma_n_star_ex = compute_sigma_n_star_ex(k_t_n_star_star,k_t_n_star,k_t_n_inv,omega_n,[c[0,0]])
        return mu_n_star_ex,sigma_n_star_ex