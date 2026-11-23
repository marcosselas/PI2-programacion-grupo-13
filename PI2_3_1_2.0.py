import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import textwrap

GAMMA = 1.4
R_AIRE = 287.0

# 1. Calibracion del sensor principal de empuje ( Voltaje [V] vs Fuerza [kN ])
voltaje_sensor = np . array ([0.0 , 1.25 , 2.45 , 3.80 , 5.10])
fuerza_real = np . array ([0.0 , 15.2 , 31.0 , 48.5 , 65.3])

# 2. Reacciones en la plataforma de lanzamiento ( Coordenadas apoyos y pesos )
# Matriz de geometria estatica ( distancias en m) para el balance de masas
matriz_estatica = np . array ([
[1.0 , 1.0 , 1.0] , # Suma de fuerzas Z
[0.0 , 1.5 , -1.5] , # Momentos en X
[ -2.0 , 1.0 , 1.0] # Momentos en Y
])
vector_cargas = np . array ([25000.0 , 1500.0 , -800.0]) # [N, Nm , Nm]

# 3. Datos de Atmosfera Estandar ( Altitud [m] , Temperatura [K] , Presion [Pa ])
h_atm = np . array ([0 , 2000 , 4000 , 6000 , 8000 , 10000 , 12000 , 14000])
T_atm = np . array ([288.15 , 275.15 , 262.15 , 249.15 , 236.15 , 223.15 , 216.65 ,
216.65])
P_atm = np . array ([101325 , 79501 , 61660 , 47217 , 35651 , 26499 , 19399 , 14170])

# 4. Telemetria de vuelo : Tiempo [s] y Velocidad [m/s]
t_vuelo = np . linspace (0 , 60 , 1200) # Muestreo a 20 Hz
# ( Perfil simulado de velocidad de un cohete )
v_vuelo = 800 * np . sin ( np . pi * t_vuelo / 60) * np . exp ( -0.02 * t_vuelo )

# --- Parametros auxiliares y funciones de apoyo (atmosfera y eje secundario) ---
VEL_ASCENSO = 300.0   # [m/s] pendiente de la aproximacion lineal h(t) = 300·t


def temperatura_y_sonido(t):
    """
    h(t) = VEL_ASCENSO·t, T(t) por interpolación 1D en la tabla atmosférica y
    velocidad del sonido a(t) = sqrt(γ·R·T(t)). Devuelve (h, T, a).
    """
    h = VEL_ASCENSO * t
    T = np.interp(h, h_atm, T_atm)
    a = np.sqrt(GAMMA * R_AIRE * T)
    return h, T, a


def preparar_eje_secundario(ax2):
    """Limpia el eje gemelo (twinx) y lo deja visible, con los ticks a la derecha."""
    ax2.cla()
    ax2.set_visible(True)
    ax2.patch.set_visible(False)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.xaxis.set_visible(False)



def balance_de_masas(A, b):
    # Resuelve el sistema lineal para encontrar las reacciones en los 3 apoyos.
    # Ax = b
    reacciones = np.linalg.solve(A, b)
    
    # Vectores de reacciones individuales
    r1, r2, r3 = reacciones
    
    # Interpretacion fisica
    interpretacion = (
        f"Las reacciones son: R1={r1:.1f} N, R2={r2:.1f} N, R3={r3:.1f} N.\n"
        "Interpretación: La carga se distribuye de manera asimétrica para contrarrestar "
        "los momentos flectores del viento, sufriendo el apoyo 2 la mayor compresión "
        "y el apoyo 3 la menor."
    )
    
    # Se imprime
    print("--- APARTADO 1: Balance de Masas ---")
    print(interpretacion)
    print("-" * 40)
    
    return reacciones, interpretacion

reacciones_apoyos, texto_ap1 = balance_de_masas(matriz_estatica, vector_cargas)


def apartado_1_balance(ax, ax2, p):
    """Apartado 1 (GUI): barras con las reacciones de los 3 apoyos y reporte."""
    r = reacciones_apoyos
    x = np.arange(r.size)
    barras = ax.bar(x, r, color=["tab:blue", "tab:orange", "tab:green"], width=0.55)
    ax.bar_label(barras, fmt="%.1f N", padding=3)
    ax.axhline(vector_cargas[0] / r.size, ls="--", color="k", lw=1,
               label=f"Reparto uniforme = {vector_cargas[0] / r.size:.0f} N")
    ax.set_xticks(x)
    ax.set_xticklabels(["Apoyo 1", "Apoyo 2", "Apoyo 3"])
    ax.set_ylim(0.0, 1.15 * r.max())
    ax.set_ylabel("Fuerza de reacción [N]")
    ax.set_title("Apartado 1 · Balance de masas (A·x = b)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=9)

    residuo = np.max(np.abs(matriz_estatica @ r - vector_cargas))
    reporte = (f"R1 = {r[0]:.1f} N\n"
               f"R2 = {r[1]:.1f} N\n"
               f"R3 = {r[2]:.1f} N\n"
               f"ΣR = {r.sum():.1f} N\n"
               f"|A·x - b|max = {residuo:.2e}")
    if p.get("imprimir", True):
        print(texto_ap1)
    return reporte, texto_ap1



#####################
# Apartado 2: Calibracion del Motor (Regresion Lineal)

def apartado_2_calibracion(ax, ax2, p):
    """
    Apartado 2: recta de mejor ajuste F(V) = m·V + c minimizando el error
    cuadrático medio respecto a los datos de calibración.
    """
    V, F = voltaje_sensor, fuerza_real
 
    # Matriz de diseño [V, 1]; lstsq resuelve min ||A·θ - F||² en un paso
    A = np.column_stack((V, np.ones_like(V)))
    (m, c), *_ = np.linalg.lstsq(A, F, rcond=None)
 
    F_aj = m * V + c                                  # fuerza ajustada
    res = F - F_aj                                    # residuos
    mse = np.mean(res ** 2)                           # error cuadrático medio
    rmse = np.sqrt(mse)
    r2 = 1.0 - np.sum(res ** 2) / np.sum((F - F.mean()) ** 2)
 
    # Predicción para el voltaje elegido en el slider
    Vq = p["v_consulta"]
    Fq = m * Vq + c
 
    V_lin = np.linspace(0.0, 5.3, 200)
    ax.vlines(V, F_aj, F, color="crimson", alpha=0.6, lw=2, label="Residuos")
    ax.scatter(V, F, color="crimson", zorder=4, label="Datos de calibración")
    ax.plot(V_lin, m * V_lin + c, color="navy", lw=2,
            label=f"F(V) = {m:.3f}·V {c:+.3f}")
    ax.plot([Vq, Vq, 0.0], [0.0, Fq, Fq], ":", color="seagreen")
    ax.scatter([Vq], [Fq], marker="D", s=90, color="seagreen", zorder=5,
               label=f"Consulta: F({Vq:.2f} V) = {Fq:.2f} kN")
    ax.set_xlabel("Voltaje del sensor piezoeléctrico [V]")
    ax.set_ylabel("Empuje [kN]")
    ax.set_title("Apartado 2 · Calibración del motor (regresión lineal)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
 
    reporte = (
        f"m (sensibilidad) = {m:.4f} kN/V\n"
        f"c (offset)       = {c:+.4f} kN\n"
        f"MSE = {mse:.4f} kN²   RMSE = {rmse:.3f} kN\n"
        f"R²  = {r2:.6f}\n"
        f"F({Vq:.2f} V) = {Fq:.3f} kN"
    )
    interp = (f"La matriz de calibración indica un sesgo de {c * 1e3:+.1f} "
              f"Newtons a 0 V (≈{abs(c) / F.max() * 100:.2f} % del fondo de "
              f"escala) y una sensibilidad de {m:.3f} kN/V; con R² = {r2:.5f} "
              f"el sensor piezoeléctrico es prácticamente lineal y {Vq:.2f} V "
              f"equivalen a {Fq:.2f} kN de empuje.")
    if p.get("imprimir", True):
        print("Apartado 2:", interp)
    return reporte, interp








"""
    Apartado 3: Perfil Atmosférico (Interpolación 1D).
    Calcula h(t) = 300*t e interpola T(t) usando la tabla atmosférica.
    """

def interpolacion(tiempo, h_ref, t_ref):

    #Calcula la temperatura en función del tiempo
    h_vuelo = 300.0 * tiempo  # Altitud estimada en m

    #Interpola T(t) usando la tabla atmosférica
    t_vuelo = np.interp(h_vuelo, h_ref, t_ref)
  

    print("=" * 70)
    print("APARTADO 3: PERFIL ATMOSFÉRICO (INTERPOLACIÓN 1D)")
    print(f"Altitud máxima estimada del modelo: {h_vuelo[-1]:.1f} m")
    print(f"Temperatura a nivel del mar (t=0 s): {t_vuelo[0]:.2f} K ({t_vuelo[0] - 273.15:.2f} °C)")
    print(f"Temperatura mínima alcanzada: {np.min(t_vuelo):.2f} K ({np.min(t_vuelo) - 273.15:.2f} °C)")
    print("Interpretación: La temperatura experimenta un gradiente térmico troposférico estándar (≈ -6.5 K/km) "
          "hasta estabilizarse en la isoterma de la tropopausa (216.65 K) " \
          "a partir de los 12.000 metros.")
    print("=" * 70)

    return h_vuelo, t_vuelo
  
h_cohete, T_cohete = interpolacion(t_vuelo, h_atm, T_atm)


def apartado_3_perfil(ax, ax2, p):
    """Apartado 3 (GUI): T(t) interpolada, tabla atmosférica y altitud h(t)."""
    t, h, T = t_vuelo, h_cohete, T_cohete
    i_trop = np.argmax(T <= T_atm.min())               # primer instante en la isoterma

    ax.plot(t, T, color="tab:red", lw=2, label="T(t) interpolada [K]")
    ax.scatter(h_atm / VEL_ASCENSO, T_atm, color="k", zorder=4,
               label="Puntos de la tabla atmosférica")
    ax.scatter([t[i_trop]], [T[i_trop]], marker="D", s=80, color="tab:blue",
               zorder=5, label=f"Inicio isoterma: t = {t[i_trop]:.1f} s")
    ax2.plot(t, h / 1e3, color="tab:gray", lw=1.2, alpha=0.8, label="h(t) [km]")
    ax2.set_ylabel("Altitud h(t) [km]", color="tab:gray")
    ax.set_xlabel("Tiempo de vuelo [s]")
    ax.set_ylabel("Temperatura ambiente T(t) [K]")
    ax.set_title("Apartado 3 · Perfil atmosférico (interpolación 1D)")
    ax.grid(alpha=0.3)
    handles, _ = ax.get_legend_handles_labels()
    l_h, = ax2.get_lines()[-1:]
    ax.legend(handles=handles + [l_h], loc="upper right", fontsize=9)

    reporte = (f"T(0 s)     = {T[0]:.2f} K ({T[0] - 273.15:.2f} °C)\n"
               f"T mín      = {T.min():.2f} K ({T.min() - 273.15:.2f} °C)\n"
               f"Tropopausa = {t[i_trop]:.1f} s (h = {h[i_trop] / 1e3:.1f} km)\n"
               f"h(60 s)    = {h[-1] / 1e3:.1f} km")
    interp = (f"La temperatura cae desde {T[0]:.1f} K a nivel del mar hasta "
              f"{T.min():.1f} K, valor que alcanza en t ≈ {t[i_trop]:.1f} s "
              f"(h ≈ {h[i_trop] / 1e3:.0f} km); por encima de la tabla "
              f"(14 km) np.interp mantiene constante el último valor.")
    if p.get("imprimir", True):
        print("Apartado 3:", interp)
    return reporte, interp



####################
# Apartado 4: Busqueda del Regimen Transonico (Ecuacion No Lineal)


def apartado_4_regimen_transonico(ax, ax2, p):
    """
    Apartado 4: instante exacto t_c en que M(t) = M_crit.
 
        M(t) = v_vuelo(t) / a(t),   a(t) = sqrt(γ·R·T(t)),   h(t) = 300·t
 
    Es una ecuación no lineal en t. Se resuelve de forma vectorizada: se evalúa
    M(t) sobre una malla muy fina, se localiza el cambio de signo de
    M(t) - M_crit y se interpola linealmente dentro del intervalo (error del
    orden de 1e-6 s con la malla utilizada).
    """
    mc = p["mach_crit"]
 
    # --- Curva de telemetría (para dibujar) ---
    h, T, a = temperatura_y_sonido(t_vuelo)
    mach = v_vuelo / a
 
    # --- Malla fina para resolver M(t) = M_crit con precisión ---
    t_f = np.linspace(t_vuelo[0], t_vuelo[-1], 200001)
    v_f = np.interp(t_f, t_vuelo, v_vuelo)
    h_f, T_f, a_f = temperatura_y_sonido(t_f)
    M_f = v_f / a_f
 
    mask = M_f >= mc                                   # tramo por encima de Mc
    alcanzado = bool(np.any(mask))
    i_max = np.argmax(M_f)
 
    ax.plot(t_vuelo, mach, color="tab:blue", lw=2, label="M(t) = v/a")
    ax.plot([t_vuelo[0], t_vuelo[-1]], [mc, mc], "--", color="tab:red",
            label=f"Mach crítico = {mc:.2f}")
 
    preparar_eje_secundario(ax2)
    l_T, = ax2.plot(t_vuelo, T, color="tab:gray", lw=1.2, alpha=0.8,
                    label="T(t) ambiente [K]")
    ax2.set_ylabel("Temperatura ambiente T(t) [K]", color="tab:gray")
 
    if alcanzado:
        i = int(np.argmax(mask))                       # primer índice con M >= Mc
        i = max(i, 1)                                  # M(0) = 0 < Mc siempre
        # Entrada: M es creciente en [i-1, i] -> interpolación directa
        t_c = np.interp(mc, M_f[i - 1:i + 1], t_f[i - 1:i + 1])
        # Salida: último índice por encima de Mc (M decreciente) -> se invierte
        j = mask.size - 1 - int(np.argmax(mask[::-1]))
        j = min(j, mask.size - 2)
        t_s = np.interp(mc, M_f[j:j + 2][::-1], t_f[j:j + 2][::-1])
 
        h_c = VEL_ASCENSO * t_c
        T_c = np.interp(h_c, h_atm, T_atm)
        a_c = np.sqrt(GAMMA * R_AIRE * T_c)
        v_c = np.interp(t_c, t_vuelo, v_vuelo)
 
        ax.plot([t_c, t_c], [0.0, mc], ":", color="tab:red")
        ax.scatter([t_c], [mc], color="tab:red", s=90, zorder=5,
                   label=f"t_c = {t_c:.3f} s")
        ax.fill_between(t_f[::200], 0, M_f[::200], where=mask[::200],
                        color="tab:red", alpha=0.12)
 
        lineas = (
            f"Mach crítico  = {mc:.2f}\n"
            f"t_c           = {t_c:.3f} s\n"
            f"h(t_c)        = {h_c / 1e3:.2f} km\n"
            f"T(t_c)        = {T_c:.2f} K\n"
            f"a(t_c)        = {a_c:.1f} m/s\n"
            f"v(t_c)        = {v_c:.1f} m/s\n"
            f"Mach máx.     = {M_f[i_max]:.2f} (t = {t_f[i_max]:.1f} s)\n"
            f"Baja de Mc en = {t_s:.2f} s"
        )
        regimen = "transónico" if mc < 1.2 else "supersónico"
        interp = (f"El cohete alcanza Mach {mc:.2f} en t_c = {t_c:.3f} s, a "
                  f"{h_c / 1e3:.2f} km de altitud donde el aire está a "
                  f"{T_c:.1f} K (a = {a_c:.0f} m/s) y vuela a {v_c:.0f} m/s; "
                  f"permanece en régimen {regimen} hasta t = {t_s:.1f} s, por "
                  f"lo que la estructura debe soportar el pico de resistencia "
                  f"de onda en ese intervalo.")
    else:
        lineas = (
            f"Mach crítico  = {mc:.2f}\n"
            f"t_c           = no alcanzado\n"
            f"Mach máx.     = {M_f[i_max]:.2f} (t = {t_f[i_max]:.1f} s)\n"
            f"Margen        = {mc - M_f[i_max]:.2f} Mach"
        )
        interp = (f"El cohete solo llega a Mach {M_f[i_max]:.2f} (t = "
                  f"{t_f[i_max]:.1f} s), por debajo del Mach crítico "
                  f"{mc:.2f}: con este perfil de velocidad la ecuación "
                  f"M(t) = {mc:.2f} no tiene solución.")
 
    ax.set_xlabel("Tiempo de vuelo [s]")
    ax.set_ylabel("Número de Mach  M(t)")
    ax.set_title("Apartado 4 · Régimen transónico: M(t) = M_crit "
                 "(mueve el slider de Mach)")
    ax.grid(alpha=0.3)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [l_T], loc="lower center", fontsize=9, ncol=4)
    if p.get("imprimir", True):
        print("Apartado 4:", interp)
    return lineas, interp



####################
# Apartado 5: Acelerometro Virtual (Derivacion Numerica)

def apartado_5_acelerometro(ax, ax2, p):
    """
    Apartado 5: aceleración longitudinal a_long(t) = dv/dt con diferencias
    finitas de segundo orden, totalmente vectorizada.

    Malla uniforme de paso dt = t[1] - t[0]:
      - Interior (centradas, O(dt^2)):  a_i = (v_{i+1} - v_{i-1}) / (2·dt)
      - Extremo inicial (adelantada, O(dt^2)):
            a_0 = (-3·v_0 + 4·v_1 - v_2) / (2·dt)
      - Extremo final (atrasada, O(dt^2)):
            a_N = ( 3·v_N - 4·v_{N-1} + v_{N-2}) / (2·dt)
    En los extremos no existe punto a ambos lados, por eso se usa la fórmula
    unilateral de segundo orden para mantener el mismo orden de precisión.
    """
    t, v = t_vuelo, v_vuelo
    dt = t[1] - t[0]                                   # paso uniforme

    a_long = np.empty_like(v)
    a_long[1:-1] = (v[2:] - v[:-2]) / (2.0 * dt)       # centradas (interior)
    a_long[0] = (-3.0 * v[0] + 4.0 * v[1] - v[2]) / (2.0 * dt)
    a_long[-1] = (3.0 * v[-1] - 4.0 * v[-2] + v[-3]) / (2.0 * dt)

    i_max = np.argmax(a_long)                          # aceleración máxima
    i_min = np.argmin(a_long)                          # deceleración máxima
    g0 = 9.80665
    cruce = np.flatnonzero(np.diff(np.sign(a_long)) != 0)  # a = 0 -> v máx.
    t_vmax = t[cruce[0]] if cruce.size else t[np.argmax(v)]

    ax.plot(t, a_long, color="tab:green", lw=2, label="a_long(t) = dv/dt")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.scatter([t[i_max]], [a_long[i_max]], color="tab:red", s=80, zorder=5,
               label=f"a máx = {a_long[i_max]:.1f} m/s² (t = {t[i_max]:.2f} s)")
    ax.scatter([t[i_min]], [a_long[i_min]], color="tab:purple", s=80, zorder=5,
               label=f"a mín = {a_long[i_min]:.1f} m/s² (t = {t[i_min]:.2f} s)")
    ax2.plot(t, v, color="tab:gray", lw=1.2, alpha=0.8, label="v(t) [m/s]")
    ax2.set_ylabel("Velocidad v(t) [m/s]", color="tab:gray")
    ax.set_xlabel("Tiempo de vuelo [s]")
    ax.set_ylabel("Aceleración longitudinal [m/s²]")
    ax.set_title("Apartado 5 · Acelerómetro virtual (diferencias finitas, O(dt²))")
    ax.grid(alpha=0.3)
    handles, _ = ax.get_legend_handles_labels()
    l2, = ax2.get_lines()[-1:]
    ax.legend(handles=handles + [l2], loc="upper right", fontsize=9)

    reporte = (
        f"dt            = {dt:.5f} s\n"
        f"a máx         = {a_long[i_max]:.2f} m/s² ({a_long[i_max] / g0:.2f} g)\n"
        f"t(a máx)      = {t[i_max]:.2f} s\n"
        f"a mín         = {a_long[i_min]:.2f} m/s² ({a_long[i_min] / g0:.2f} g)\n"
        f"t(a mín)      = {t[i_min]:.2f} s\n"
        f"a(t=0)        = {a_long[0]:.2f} m/s²\n"
        f"a = 0 (v máx) = {t_vmax:.2f} s"
    )
    interp = (f"El acelerómetro virtual registra un pico de "
              f"{a_long[i_max]:.1f} m/s² ({a_long[i_max] / g0:.2f} g) en "
              f"t = {t[i_max]:.2f} s; la aceleración se anula en "
              f"t ≈ {t_vmax:.1f} s (velocidad máxima) y después el cohete "
              f"decelera hasta {a_long[i_min]:.1f} m/s² por resistencia "
              f"aerodinámica y gravedad.")
    print("=" * 70)
    print("APARTADO 5: ACELERÓMETRO VIRTUAL (DERIVACIÓN NUMÉRICA)")
    print(reporte)
    print("Interpretación:", interp)
    print("=" * 70)
    return reporte, interp, a_long



####################
# Apartado 6: Altitud Apogeo (Integracion Numerica)

def apartado_6_apogeo(ax, ax2, p):
    """
    Apartado 6: h_total = ∫_0^60 v(t) dt con la regla del trapecio compuesta,
    vectorizada con arrays:

        h_total = Σ_{i=0}^{N-1} (v_i + v_{i+1}) / 2 · (t_{i+1} - t_i)

    Se calcula también la altitud acumulada h(t) con np.cumsum.
    """
    t, v = t_vuelo, v_vuelo
    tramos = 0.5 * (v[1:] + v[:-1]) * np.diff(t)       # área de cada trapecio
    h_total = np.sum(tramos)                           # altitud a t = 60 s
    h_acum = np.concatenate(([0.0], np.cumsum(tramos)))  # h(t) acumulada

    # Referencia analítica de este perfil: ∫ e^{αt}·sin(βt) dt cerrada
    al, be = -0.02, np.pi / 60.0
    h_exacta = 800.0 * be * (1.0 + np.exp(al * 60.0)) / (al ** 2 + be ** 2)
    err_rel = abs(h_total - h_exacta) / h_exacta * 100.0

    ax.plot(t, v, color="tab:blue", lw=2, label="v(t) [m/s]")
    ax.fill_between(t, 0.0, v, color="tab:blue", alpha=0.15,
                    label="Área bajo la curva")
    ax2.plot(t, h_acum / 1e3, color="tab:orange", lw=2,
             label="h(t) acumulada [km]")
    ax2.set_ylabel("Altitud acumulada [km]", color="tab:orange")
    ax.set_xlabel("Tiempo de vuelo [s]")
    ax.set_ylabel("Velocidad [m/s]")
    ax.set_title("Apartado 6 · Altitud de apogeo (regla del trapecio)")
    ax.grid(alpha=0.3)
    handles, _ = ax.get_legend_handles_labels()
    l2, = ax2.get_lines()[-1:]
    ax.legend(handles=handles + [l2], loc="upper right", fontsize=9)

    reporte = (
        f"h_total (trapecio) = {h_total:.2f} m ({h_total / 1e3:.3f} km)\n"
        f"h_total (analítica)= {h_exacta:.2f} m\n"
        f"Error relativo     = {err_rel:.4f} %"
    )
    interp = (f"Integrando la velocidad con la regla del trapecio, el cohete "
              f"alcanza {h_total / 1e3:.2f} km a t = 60 s (error "
              f"{err_rel:.3f} % frente a la integral exacta del perfil).")
    print("=" * 70)
    print("APARTADO 6: ALTITUD APOGEO (INTEGRACIÓN NUMÉRICA)")
    print(reporte)
    print("Interpretación:", interp)
    print("=" * 70)
    return reporte, interp



####################
# Dashboard interactivo: una unica ventana con RadioButtons + Sliders + reporte

ETIQUETAS = ["1 · Balance de masas", "2 · Calibración del motor",
             "3 · Perfil atmosférico", "4 · Régimen transónico",
             "5 · Acelerómetro virtual", "6 · Altitud apogeo"]

FUNCIONES = {1: apartado_1_balance, 2: apartado_2_calibracion,
             3: apartado_3_perfil, 4: apartado_4_regimen_transonico,
             5: apartado_5_acelerometro, 6: apartado_6_apogeo}


def construir_dashboard():
    """Crea la figura única con todos los widgets y devuelve (fig, widgets)."""
    params = {"mach_crit": 1.2, "v_consulta": 3.0, "imprimir": True}

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Telemetría y dinámica de vuelo · cohete sonda AeroPy-I",
                 fontsize=14, fontweight="bold")
    ax = fig.add_axes([0.35, 0.37, 0.57, 0.53])         # eje principal
    ax2 = ax.twinx()                                    # eje secundario
    ax_radio = fig.add_axes([0.02, 0.55, 0.25, 0.35])   # panel izquierdo: RadioButtons
    ax_rep = fig.add_axes([0.02, 0.05, 0.25, 0.42])     # reporte numérico
    ax_int = fig.add_axes([0.35, 0.03, 0.61, 0.15])     # interpretación
    ax_sl_mach = fig.add_axes([0.44, 0.26, 0.42, 0.03])
    ax_sl_volt = fig.add_axes([0.44, 0.26, 0.42, 0.03])

    ax_radio.set_title("Apartado", fontsize=11)
    ax_rep.set_title("Reporte numérico", fontsize=11)
    ax_int.set_title("Interpretación", fontsize=11)
    ax_rep.set_xticks([]); ax_rep.set_yticks([])
    ax_int.set_xticks([]); ax_int.set_yticks([])
    txt_rep = ax_rep.text(0.04, 0.96, "", va="top", family="monospace",
                          fontsize=8.5, transform=ax_rep.transAxes)
    txt_int = ax_int.text(0.015, 0.92, "", va="top", fontsize=10,
                          transform=ax_int.transAxes)

    radio = RadioButtons(ax_radio, ETIQUETAS, active=0)
    sl_mach = Slider(ax_sl_mach, "Mach crítico", 0.8, 2.5, valinit=1.2, valstep=0.01)
    sl_volt = Slider(ax_sl_volt, "V consulta [V]", 0.0, 5.3, valinit=3.0, valstep=0.05)

    def redibujar():
        n = ETIQUETAS.index(radio.value_selected) + 1
        ax_sl_mach.set_visible(n == 4)                  # cada slider solo en su apartado
        ax_sl_volt.set_visible(n == 2)
        ax.cla()
        if n in (1, 2):
            ax2.cla()
            ax2.set_visible(False)
        else:
            preparar_eje_secundario(ax2)
        salida = FUNCIONES[n](ax, ax2, params)
        txt_rep.set_text(salida[0])
        txt_int.set_text(textwrap.fill(salida[1], 105))
        fig.canvas.draw_idle()

    def al_elegir(_):
        params["imprimir"] = True                       # imprime en terminal al cambiar de apartado
        redibujar()

    def al_mover(_):
        params["mach_crit"] = sl_mach.val
        params["v_consulta"] = sl_volt.val
        params["imprimir"] = False                      # sin spam en terminal al arrastrar
        redibujar()

    radio.on_clicked(al_elegir)
    sl_mach.on_changed(al_mover)
    sl_volt.on_changed(al_mover)
    redibujar()
    return fig, (radio, sl_mach, sl_volt)


if __name__ == "__main__":
    fig_dash, widgets_dash = construir_dashboard()      # se guardan las referencias
    plt.show()
