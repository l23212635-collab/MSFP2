"""
Práctica 2: Sistema cardiovascular
Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: Perez Garcia Diego
Número de control: 23212635
Correo institucional: l23212635@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt
import control as ctrl
import pandas as pd
import scipy.signal as signal_scipy

# Datos de la simulación
u_excel = np.array(pd.read_excel('signal.xlsx', header=None))
u = np.reshape(u_excel,-1)
x0, t0, tend, dt, w, h = 0, 0, 15, 1E-3, 10,5
N_steps = round(tend/dt) + 1
t = np.linspace(t0, tend, N_steps)
u = np.reshape(signal_scipy.resample(u_excel, len(t)), -1).real

signals = ['Cardiovascular_Lazo abierto']


Z, R, L = 0.010, 1.000, 0.001
C_normo = 1.500
C_hipo  = 3.500
C_hiper = 1.000

num_comun = [L * R, R * Z]
den_normo = [C_normo * L * R * Z, L * (Z + R), R * Z]
den_hipo  = [C_hipo * L * R * Z,  L * (Z + R), R * Z]
den_hiper = [C_hiper * L * R * Z, L * (Z + R), R * Z]

sys_normo = ctrl.tf(num_comun, den_normo)
sys_hipo  = ctrl.tf(num_comun, den_hipo)
sys_hiper = ctrl.tf(num_comun, den_hiper)

print(f"Sistema Normotenso (Lazo Abierto):\n{sys_normo}\n")
print(f"Sistema Hipotenso (Lazo Abierto):\n{sys_hipo}\n")
print(f"Sistema Hipertenso (Lazo Abierto):\n{sys_hiper}\n")

# PID
P_hipo, I_hipo, D_hipo, N_hipo = 47.1616, 11761.4928, 0.0058764, 5134.0938
P_hiper, I_hiper, D_hiper, N_hiper = 10.6371, 3103.2143, 0.0018407, 3062.3768

PID_hipo = ctrl.tf([P_hipo + D_hipo*N_hipo, P_hipo*N_hipo + I_hipo, I_hipo*N_hipo], [1, N_hipo, 0])
PID_hiper = ctrl.tf([P_hiper + D_hiper*N_hiper, P_hiper*N_hiper + I_hiper, I_hiper*N_hiper], [1, N_hiper, 0])

x_hipo = ctrl.series(PID_hipo, sys_hipo)
sysPID_hipo = ctrl.feedback(x_hipo, 1, sign=-1)

x_hiper = ctrl.series(PID_hiper, sys_hiper)
sysPID_hiper = ctrl.feedback(x_hiper, 1, sign=-1)


clr1 = np.array([118, 196, 87])/255 
clr2 = np.array([27, 44, 193])/255 
clr3 = np.array([221, 113, 133])/255   

def openloop(t, sys, u_signal):

    _, PAu = ctrl.forced_response(sys, t, u_signal, x0)
    return PAu

def closedloop(t, sysPID, u_signal):
    _, PIDu = ctrl.forced_response(sysPID, t, u_signal, x0)
    return PIDu


for i in range(len(signals)):
    # Señal real obtenida de Excel
    y_normo = openloop(t, sys_normo, u)
    y_hipo  = openloop(t, sys_hipo, u)
    y_hiper = openloop(t, sys_hiper, u)
    
    y_PID_hipo  = closedloop(t, sysPID_hipo, u)
    y_PID_hiper = closedloop(t, sysPID_hiper, u)
    
    # Formato de fuentes institucional para reportes de laboratorio
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

   # Lazo abierto
    fg1 = plt.figure(1)
    fg1.set_size_inches(w, h)
    plt.plot(t, y_normo, '-', color=clr1, label='Normotenso')
    plt.plot(t, y_hipo, '--', color=clr2, label='Hipotenso')
    plt.plot(t, y_hiper, ':', color=clr3, label='Hipertenso')
    
    plt.xlim(0, 15); plt.xticks(np.arange(0, 16, 1))
    plt.ylim(-0.4, 1.2); plt.yticks(np.arange(-0.4 , 1.2, 0.2))
    plt.xlabel('t [s]')
    plt.ylabel('Respuesta [V]')
    plt.legend(bbox_to_anchor=(0.5, -0.25), loc='center', ncol=3, frameon=False)
    plt.show()
    fg1.savefig(f"{signals[i]}_.pdf", bbox_inches='tight')

  # Hipotenso
    fg2 = plt.figure(2)
    fg2.set_size_inches(w, h)
    plt.plot(t, y_normo, '-', color=clr1, label='Normotenso')
    plt.plot(t, y_hipo, '--', color=clr2, label='Hipotenso LA')
    plt.plot(t, y_PID_hipo, ':', linewidth=2.5, color=clr3, label='Hipotenso PID')
    
    plt.xlim(0, 15); plt.xticks(np.arange(0, 16, 1))
    plt.ylim(-0.4, 1.2); plt.yticks(np.arange(-0.4 , 1.2, 0.2))
    plt.xlabel('t [s]')
    plt.ylabel('Respuesta [V]')
    plt.legend(bbox_to_anchor=(0.5, -0.25), loc='center', ncol=3, frameon=False)
    plt.show()
    fg2.savefig(f"{signals[i]}_Hipotenso.pdf", bbox_inches='tight')

  # Hipertenso
    fg3 = plt.figure(3)
    fg3.set_size_inches(w, h)
    plt.plot(t, y_normo, '-', color=clr1, label='Normotenso')
    plt.plot(t, y_hiper, '--', color=clr2, label='Hipertenso LA')
    plt.plot(t, y_PID_hiper, ':', linewidth=2.5, color=clr3, label='Hipertenso PID')
    
    plt.xlim(0, 15); plt.xticks(np.arange(0, 16, 1))
    plt.ylim(-0.4, 1.2); plt.yticks(np.arange(-0.4 , 1.2, 0.2))
    plt.xlabel('t [s]')
    plt.ylabel('Respuesta [V]')
    plt.legend(bbox_to_anchor=(0.5, -0.25), loc='center', ncol=3, frameon=False)
    plt.show()
    fg3.savefig(f"{signals[i]}_Hipertenso.pdf", bbox_inches='tight')





