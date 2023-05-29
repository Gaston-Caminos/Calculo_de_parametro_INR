# -*- coding: utf-8 -*-
"""
Created on Sun May 28 14:03:07 2023

@author: Gaston Caminos
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import hilbert

#%% Tiempos de reverberacion

def TRs(RI, fs):
    """
    Calcula los tiempos de reverberación T20, T30 y T60 de una respuesta al impulso (RI).

    Args:
        RI (array_like): La respuesta al impulso.
        fs (float): La tasa de muestreo de la respuesta al impulso en Hz.

    Returns:
        tuple: Una tupla que contiene los tiempos de reverberación T20, T30 y T60,
               y una versión suavizada de la curva de la respuesta al impulso.

    """

    # Calculo el vector temporal
    time_axis = np.arange(len(RI)) / fs

    
    # Realizo un suavizado de la curva
    suavizado = suavizar_RI(RI, fs)  
    suavizado = 10 * np.log10(suavizado) #Paso a dB
    # Calculo el rango de decaimiento para cada TR
    start_index_20 = np.argmax(suavizado <= -5)
    end_index_20 = np.argmax(suavizado <= -25)
    start_index_30 = np.argmax(suavizado <= -5)
    end_index_30 = np.argmax(suavizado <= -35)

    # Extraigo el rango de tiempo correspondiente para la regresion
    time_range_20 = time_axis[start_index_20:end_index_20]
    time_range_30 = time_axis[start_index_30:end_index_30]

    # Realizo la regresion lineal para estimar los tiempos de decaimiento
    slope_20, intercept_20, _, _, _ = linregress(time_range_20, suavizado[start_index_20:end_index_20])
    slope_30, intercept_30, _, _, _ = linregress(time_range_30, suavizado[start_index_30:end_index_30])
    slope_60, intercept_60, _, _, _ = linregress(time_axis, suavizado)

    # Calculo los tiempos de reverberacion
    T20 = round(-20 / slope_20, 2)
    T30 = round(-30 / slope_30, 2)
    T60 = round(-60 / slope_60, 2)

    return T20, T30, T60, suavizado

#%% Integral inversa de Schroeder

def schroeder_int(IR):
        """
        Calcula la integral inversa de Schroeder de una respuesta al impulso (RI).
    
        Args:
            IR (array_like): La respuesta al impulso.
    
        Returns:
            array_like: La integral inversa de Schroeder de la respuesta al impulso.
    
        """
        
        sch = np.cumsum(IR[::-1])
        sch /= np.max(sch)
        
        return sch[::-1]
    
#%% Funciones de orden superior

# Suavizado de la RI

def suavizar_RI(RI, fs):
    """
    Realiza un suavizado de la respuesta al impulso (RI) mediante la transformada de Hilbert y la integral de Schroeder.

    Args:
        RI (array_like): La respuesta al impulso.
        fs (float): La frecuencia de muestreo de la respuesta al impulso.

    Returns:
        array_like: La respuesta al impulso suavizada.

    """
    
    transformada_hilbert = np.abs(hilbert(RI))      # Calculo la envolvente de mi RI con el valor absoluto de la Transformada de Hilbert
    schroeder = schroeder_int(transformada_hilbert) # Calculo la integral inversa de Schroeder
    
    return schroeder

def reemplaza_ceros(array):
    """
    Reemplaza los ceros de un array por el mínimo valor distinto de cero.

    Args:
        array (array_like): El array que se desea modificar.

    Returns:
        array_like: El array modificado con los ceros reemplazados.

    """
    
    min_nonzero = np.min(array[np.nonzero(array)])   # Encuentro el minimo valor distinto de cero
    array = np.where(array == 0, min_nonzero, array) # Reemplazo los ceros por el minimo valor distinto de cero
    
    return array

#%% Calculo del INR

def calc_INR(RI, fs, tipo_estimulo): 
    """
    Calcula el parámetro INR (Impulse to Noise Ratio) dada una respuesta al impulso (RI).

    Args:
        RI (array_like): La respuesta al impulso.
        fs (float): La frecuencia de muestreo de la RI.
        tipo_estimulo (str): El tipo de estímulo utilizado para registrar la RI. Puede ser "sweep" o "impulso".

    Returns:
        float: El valor del parámetro INR.

    """

    T20, T30, T60, RI_dB = TRs(RI, fs)
    h0 = np.max(RI) # Calculo el maximo de mi RI
    ruido = RI[int(0.8 * len(RI)): len(RI)]              # Tomo el ultimo 20% de la senal
    LN = 10 * np.log10(np.sqrt(np.mean(ruido**2)))       # Calculo el nivel de ruido
    S0 = 10 * np.log10((T60 * h0**2) / (6 * np.log(10))) # Calculo S(0)
    LIR = S0 + 10 * np.log10((6 * np.log(10)) / T60)     # Calculo LIR
    INR = LIR - LN                                       # Calculo INR
    
    graficar(RI, RI_dB, LIR, LN, tipo_estimulo) # Realizo graficos de las curvas

    return round(INR, 2)

#%% Funcion que grafica las curvas ETC y RI [dB]

def graficar(RI, RI_dB, LIR, LN, tipo_estimulo):
    """
    Realiza un gráfico de la respuesta al impulso (RI) en dB y su ETC.
    
    Args:
        RI (array_like): La respuesta al impulso.
        RI_dB (array_like): La respuesta al impulso en dB.
        LIR (float): El valor del parámetro LIR.
        LN (float): El valor del parámetro LN.
        tipo_estimulo (str): El tipo de estímulo utilizado para registrar la RI. Puede ser "sweep" o "impulso".

    Returns:
        None
    """
    
    INR = round(LIR - LN, 2)
    
    RI_max = np.argmax(np.abs(RI))  # Find index of maximum absolute value
    t = np.linspace(0, len(RI)/fs, len(RI), endpoint=False) - RI_max
    RI = RI[int(RI_max - 5):]
    RI = reemplaza_ceros(RI) # Reemplazo los ceros por el minimo valor distinto de cero para evitar problemas con el log
    RI_dB = RI_dB[int(RI_max - 5):]
     
    # Gráficos
    plt.figure(figsize=(8, 4.5), dpi=150)
    RI2 = 10 * np.log10(RI)
    #t = t[int(RI_max - 5):]
    t = t[int(RI_max - 5):] - t[int(RI_max - 5)]
    if tipo_estimulo == "sweep":
        t_limite = int(2 * fs) # Limito el vector temporal
        t = t[:t_limite]
        RI2 = RI2[:t_limite]
        RI_dB = RI_dB[:t_limite]
        
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, RI2, color="lightgray")
    plt.plot(t, RI_dB, color="gray", label='ETC', linewidth=1.5)
    plt.hlines(LIR, t[0], t[-1], color="black", linestyle=":", label='LIR', zorder=5)
    plt.hlines(LN, t[0], t[-1], color="black", linestyle="dashed", label='LN', zorder=5)
    plt.annotate("", xy=(np.median(t), LN), xytext=(np.median(t), LIR), arrowprops=dict(arrowstyle='<->'))
    plt.text(np.median(t), (LIR + LN) / 2, f'INR={INR} dB', ha='left', va='center')
    plt.ylabel('Nivel [dB]')
    plt.xlabel('Tiempo [s]')
    plt.ylim(np.nanmin(RI2[RI2 != -np.inf]) - 10, 5)
    plt.grid()
    plt.legend(loc='lower left', fancybox=True, shadow=True)

    
    

#%% Procesamiento


def procesamiento():
    audios = ["IR_Earthworks_sweep.wav", "IR_Earthworks_imp.wav", "IR_Juan_sweep.wav", "IR_Juan_imp.wav", "IR_yo_sweep.wav", "IR_yo_imp.wav", "IR_Notebook_sweep.wav", "IR_Notebook_imp.wav"]
    
    INRs = []
    for audio in audios:
        IR, fs = sf.read(audio)
        nombre_archivo = "Micrófono: "
        tipo_estimulo = "impulso"
        if "sweep" in audio:
            tipo_estimulo = "sweep"
            if "Earthworks" in audio:
                nombre_archivo += "Earthworks-sweep"
            elif "Juan" in audio:
                nombre_archivo += "Samsung s22+-sweep"
            elif "yo" in audio:
                nombre_archivo += "Celular Xiaomi-sweep"
            else:
                nombre_archivo += "PC Acer-sweep"
        else:
            if "Earthworks" in audio:
                nombre_archivo += "Earthworks-impulso"
            elif "Juan" in audio:
                nombre_archivo += "Samsung s22+-impulso"
            elif "yo" in audio:
                nombre_archivo += "Celular Xiaomi-impulso"
            else:
                nombre_archivo += "PC Acer-impulso"
        IN = calc_INR(IR, fs, tipo_estimulo)
        plt.title(f"ETC y parámetro INR para {nombre_archivo}")
        INRs.append(IN)
        
if __name__ == "__main__":
    
    procesamiento()