import numpy as np
from scipy.stats import norm

'''
On veux savoir combien vaux aujourd'hui un call C0 pour une action dont la valeur St va 
changer dans le futur et on pourrais comme pour Monte-Carlo faire une simulation des 
possibles valeur St et en faire une moyenne mais tous ces futues ne sont pas équiprobables
et suivent une lois normale, une probalité en cloche
'''

'''
ou est ce qu'on se positionne dans cette probabilité en cloche des valeur possible, (aulieux 
de faire plusieur simulation)

d1 : mesure à quel point l'option est interressantes, forte d1, forte probaliblité de 
gain, petit d1, option peu interssante (optimiste)

d2 : probabilité ajusté que l'option soit interresssante(realiste)
'''
def compute_d1_d2(S0, K, r, sigma, T):
    A = (r + ((sigma ** 2) / 2)) * T
    sigma_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + A) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    return (d1, d2)

'''
Donne la valeur C0

arguments:
S0     # prix actuel de l'actif
K      # strike (prix d'exercice)
r      # taux sans risque
sigma  # volatilité
T      # maturité (temps)
'''
def bs_call_price(S0, K, r, sigma, T):
    d1, d2 = compute_d1_d2(S0, K, r, sigma, T)
    # calcule la valeur actuelle du gain potentiel - cout futur moyen actualisé
    #norm.cdf(d1): “quelle proportion de la cloche est à gauche de d1
    C0 = (S0 * norm.cdf(d1)) - (K * np.exp(- r * T) * norm.cdf(d2))
    return C0

'''
les Greeks désigne la sensibilité du prix de l'option
delta est le pourcentage que l'on doit possédé d'une action pour réussir à ce couvir
d'un risque localement (d'ou le taux sans risque)

Cette fonction calcule:
delta : de combien doit monte le prix de call C0 pour une valeur S0 action
gamma : de combien change la valeur de delta pour le nouveau S0
vega : si sigma change de combien change delta ?
'''
def bs_greeks(S0, K, r, sigma, T):
    d1, d2 = compute_d1_d2(S0, K, r, sigma, T)
    #calcule de la dérivé partielle du prix par rapport à S0
    delta = norm.cdf(d1)
    gamma =  (norm.pdf(d1)) / (S0 * sigma * np.sqrt(T))
    vega = S0 * norm.pdf(d1) * np.sqrt(T)
    return (delta, gamma, vega)

test = [
    [100, 100, 0.05, 0.2, 1.0],
    [100, 110, 0.05, 0.2, 1.0],
    [100, 90, 0.05, 0.2, 1.0]
]
print ("test bs_call_price")
for i in test:
    print(bs_call_price(*i))

'''
resultat attendu:
10.450583572185565
6.040088129724239
16.699448408416004
'''

print ("test bs_greeks")
print(bs_greeks(100, 100, 0.05, 0.2, 1))

'''
resultat attendu:
delta : 0.6368
'''
