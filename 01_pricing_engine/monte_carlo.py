import numpy as np
import time

'''
Donne une valeur actualisée de l'espérance de tous les possibles 
CT obtenue (C0) à partir de la formule du mouvement brownien et 
de la simulation des valeur de WT le bruit du marché

arguments:
S0     # prix actuel de l'actif
K      # strike (prix d'exercice)
r      # taux sans risque
sigma  # volatilité
T      # maturité (temps)
n_paths# normbre de valeur de Wt simulées
'''
def mc_call_price(S0, K, r, sigma, T, n_paths=100000):

    #1. simulation du mouvement brownien des n_paths valeurs de ST, WT
    WT = np.random.standard_normal(size=n_paths) * np.sqrt(T)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * WT)#mouvenement brownien

    #2. Calcule de la valeur de CT pour chaque ST
    payoffs = np.maximum(ST - K, 0)#formule du payoff

    #3. Retourne la moyenne actualisée
    mean = np.exp(-r * T) * np.mean(payoffs)#formule de l'esperance * formule de l'actualisation

    return mean

np.random.seed(42)
print ("test mc_call_price")
print(mc_call_price(100, 100, 0.05, 0.2, 1))

'''
resultat attendu:
 Tu ne tomberas pas exactement sur 10.45 — le Monte Carlo est aléatoire.
Mais tu dois être proche à ±0.1 environ.
'''

'''
option exotique
Une option call à barrière knock-out : identique à un call normal, sauf que si l'action touche un niveau BB
B à n'importe quel moment avant TT
T, l'option est annulée — elle vaut 0.

n_steps=252 représente les 252 jours de bourse dans une année — on simule un point par jour.
'''
def mc_barrier_call_price(S0, K, B, r, sigma, T, n_paths=100000, n_steps=252):

    #1. Simuler des trajectoire complètes
    delta_t = T / n_steps
    #Simuler les Z ~ N(0,1)
    Z = np.random.standard_normal((n_paths, n_steps))
    # Initialisation des trajectoires
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    # Construction des trajectoires
    X = np.exp((r - 0.5 * sigma ** 2) * delta_t + sigma * np.sqrt(delta_t) * Z)
    paths[:, 1:] = S0 * np.cumprod(X, axis=1)

    # 2. vérifier la barrière
    #cree une ligne de longueur paths true/false si la valeur finale de ST depasse B
    knock_out = np.any(paths >= B, axis=1)

    #3. payoff finale
    ST = paths[:, -1]
    payoff =  np.where(knock_out, 0, np.maximum(ST - K, 0))

    # 4. actualisation
    price = np.exp(-r * T) * np.mean(payoff)

    return price

print ("test mc_barrier_call_price")
print(mc_barrier_call_price(100, 100, 120, 0.05, 0.2, 1))

'''
resultat attendu:
Le résultat attendu est autour de
6.50 — bien inférieur au call normal (10.45) car la barrière annule beaucoup de trajectoires favorables.
'''

print(mc_barrier_call_price(100, 100, 150, 0.05, 0.2, 1))
print(mc_call_price(100, 100, 0.05, 0.2, 1))

np.random.seed(42)
start = time.time()
for _ in range(10):
    mc_barrier_call_price(100, 100, 120, 0.05, 0.2, 1)
print(f"Vectorisé : {(time.time()-start)/10*1000:.1f}ms par appel")