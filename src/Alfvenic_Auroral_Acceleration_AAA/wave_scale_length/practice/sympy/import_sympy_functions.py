import spaceToolsLib as stl
import time
start_time = time.time()

stl.prgMsg('Importing Plasma Environmental Functions')
from src.Alfvenic_Auroral_Acceleration_AAA.wave_scale_length.practice.sympy.verify_sympy_functions import func_pDD_mu_V_A
stl.Done(start_time)


mu0 = -0.3233202914138799
chi0 = 0.12047

for i in range(1000):
    print(func_pDD_mu_V_A(mu0,chi0))