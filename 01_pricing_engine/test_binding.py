import os
os.add_dll_directory(r"C:\Users\bayih\AppData\Local\Programs\Python\Python312")
os.add_dll_directory(r"C:\Users\bayih\AppData\Local\Programs\Python\Python312\libs")

import sys
sys.path.append(r"C:\Users\bayih\PycharmProjects\QuantDevProject\QuantForge\01_pricing_engine")

import mc_pricer
import time

#Test prix
price = mc_pricer.mc_call_price(100, 100, 0.05, 0.2, 1.0, 100000)
print(f"Prix : {price:.4f}")

#Test vitesse
start = time.time()
for _ in range(10):
    mc_pricer.mc_call_price(100, 100, 0.05, 0.2, 1.0, 100000)
ms = (time.time() - start) / 10 * 1000
print(f"Temps moyen : {ms:.1f} ms par appel")