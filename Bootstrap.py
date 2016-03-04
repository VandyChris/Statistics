# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:17:44 2015

@author: Chenzhao
"""
import numpy as np
from scipy.stats import t

class bootstrap():        
    def __init__(self, sample, f):
        '''
        bootstrap to compute confidence intervals
        Reference: Singh, Kesar, and Minge Xie. "Bootstrap: a statistical method." Rutgers University, USA. Retrieved from http://www. stat. rutgers. edu/home/mxie/RCPapers/bootstrap. pdf (2008).
        '''
        self.sample = sample # user-defined sample, (n,) array
        self.f = f # the function to compute the desired statistic, f(sample) = theta
        self.n = np.shape(sample)[0] # sample size
        self.theta = f(sample) # the statistic computed by the given sample
        self.btheta = self.getbsample() # the statistic computed by the bootstrap sample sets
        
        # three different 95% confidence intervals (CI)
        self.CI_percentile = (None, None)
        self.CI_centered = (None, None)
        self.CI_t = (None, None)
        
    def getbsample(self):
        '''
        Generate N bootstrap samples of theta 
        '''
        N = min(int(self.n*np.log(self.n)), 10000) # number of generated bootstrap sample sets
        index = np.random.randint(self.n, size = (N, self.n))
        btheta = np.zeros((N,))
        for i in range(N):
            bsample = self.sample[index[i, :]]
            btheta[i] = self.f(bsample)
        return btheta
            
    def getCI_percentile(self):
        '''
        the simplest bootstrap, lower bound = 2.5 percentile and upper bound = 97.5 percentile
        '''
        self.CI_percentile = (np.percentile(self.btheta, 2.5), np.percentile(self.btheta, 97.5))
        return self.CI_percentile
        
    def getCI_centered(self):
        '''
        centerned confidence bounds
        '''
        self.CI_centered = (self.theta*2 - np.percentile(self.btheta, 97.5), self.theta*2 - np.percentile(self.btheta, 2.5))
        return self.CI_centered
    
    def getCI_t(self):
        '''
        the t-method confidence bounds, accurator than earlier methods
        '''
        SE = np.std(self.btheta, ddof = 1)
        self.CI_t = (self.theta - SE*t.ppf(0.975, self.n - 1), self.theta - SE*t.ppf(0.025, self.n - 1))
        return self.CI_t

if __name__ == '__main__':
    # bootstrapping to compute the confidence bounds of mean value
    sample = np.random.normal(loc = 4, scale = 2, size = 1000)
    test = bootstrap(sample, np.mean)
    print('The percential confidence bounds of the mean value is:')
    print(test.getCI_percentile())
    print('\n')
    
    print('The centered confidence bounds of the mean value is:')
    print(test.getCI_centered())
    print('\n')
    
    print('The t-confidence bounds of the mean value is:')
    print(test.getCI_t())
    print('\n')
