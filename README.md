# State space modelling of infant movements

**Code repository for Passmore et al. 2024**

We applied autoregressive Hidden Markov Models (ARHMM) to joint positions extracted from smartphone videos of 12-14 week old infants.


![graphical abstract](/img/abstract.png)  


see notebooks for details:

[**PrincipalMovements**](/PrincipalMovements.ipynb)  
Initial processing and decomposition of movement data into a set of ['principal movements'](https://www.sciencedirect.com/science/article/abs/pii/S0021929015007381)

[**ARHMM**](/ARHMM.ipynb)  
State-space modelling of keypoint trajectories with 5-fold cross-validation.

[**Analysis**](/Analysis_and_Figures.ipynb)  
Visualisation and statistical analysis of movement states
