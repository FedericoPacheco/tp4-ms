import matplotlib.pyplot as mplplt
import matplotlib.image as mplimg
import numpy as np
import math
import colorsys
from timeit import default_timer as timer

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Constantes

# Colores de las zonas de la imagen
LIQUIDO     = np.array([154/255, 194/255, 230/255]) # Celeste
PAREDES     = np.array([254/255,   0/255,   0/255]) # Rojo
INTERIOR    = np.array([255/255, 127/255,   0/255]) # Naranja

MUESTRA     = np.array([  0/255,   0/255,   0/255]) # Negro

# Temperaturas, en °C
T0_LIQUIDO      = -150.
T0_PAREDES      = 35.
T0_INTERIOR     = 37.
T_BORDE         = -150. # Se supone constante e igual alrededor de toda la malla
T_ENFRIAMIENTO  = 0.

# Difusividad, en mm^2/s
K_LIQUIDO   = 1.
K_PAREDES   = 0.5
K_INTERIOR  = 0.8

# Valores por defecto
K_DFLT  = 0.
T_DFLT  = -273.15
HT_DFLT = 1.
HX_DFLT = 0.5
HY_DFLT = 0.5

# Otros
kT0List = [(LIQUIDO, K_LIQUIDO, T0_LIQUIDO), (PAREDES, K_PAREDES, T0_PAREDES), (INTERIOR, K_INTERIOR, T0_INTERIOR)] 
e = 1e-4

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Diferencias finitas:

# hx: paso horizontal en el espacio, en mm  ; x = x0 + j * hx
# hy: paso vertical en el espacio, en mm    ; y = y0 + i * hy
# ht: paso en el tiempo
# t0: tiempo inicial
# tf: tiempo final
# imgCorMuest: imagen original marcada con puntos donde se tomaron las muestras
# imgCorDiscret: version reducida / discretizada de la imagen original

def discretizar(imgCor, hx = HX_DFLT, hy = HY_DFLT):

    rows, cols, _ = np.shape(imgCor)
    hi = mmAPixel(hy)
    hj = mmAPixel(hx)

    rowsD = math.ceil(rows / hi - e)
    colsD = math.ceil(cols / hj - e)    
    
    imgCorMuest = np.copy(imgCor)
    imgCorDiscret = np.full((rowsD, colsD, 3), LIQUIDO, dtype = np.float64)
    
    i0 = math.floor(hi / 2 + e)
    j0 = math.floor(hj / 2 + e)
    for i in range (0, rowsD): 
        for j in range (0, colsD):

            ii = i0 + i * hi
            jj = j0 + j * hj

            if ii < rows and jj < cols:
                imgCorMuest[ii][jj] = MUESTRA
                imgCorDiscret[i][j] = imgCor[ii][jj]
                
    return imgCorMuest, imgCorDiscret

def mapearKyT0(imgCorDiscret):
    
    rowsD, colsD, _ = np.shape(imgCorDiscret)
    kArrArr = np.full((rowsD, colsD), K_DFLT)
    T0ArrArr = np.full((rowsD, colsD), T_DFLT)

    for i in range (0, rowsD): 
        for j in range (0, colsD):
            k, T0 = getKT0(imgCorDiscret[i][j])
            kArrArr[i][j] = k
            T0ArrArr[i][j] = T0
           
    return kArrArr, T0ArrArr

def getKT0(zonaEval):
    kRes = K_DFLT
    T0Res = T_DFLT
    for i in range(len(kT0List)):
        zona, k, T0 = kT0List[i]
        #np.linalg.norm(zonaEval - zona, np.inf) < e:
        if np.isclose(zonaEval, zona, e)[0]: 
            kRes = k
            T0Res = T0

    return kRes, T0Res
  

def diferenciasFinitas(imgCorDiscret, t0, tf, ht = HT_DFLT, hx = HX_DFLT, hy = HY_DFLT):

    tArr = [t0]
   
    k, T0 = mapearKyT0(imgCorDiscret)
    rows, cols = np.shape(T0)
    TArrArrArr = [T0]

    corCaliente = True 
    TLimInf = np.full((rows, cols), T_ENFRIAMIENTO)    
    tEnfri = np.inf

    # Iterar hasta que el corazon este frio o se llegue a tf, lo que ocurra primero
    l = 1
    maxIter = math.ceil((tf - t0) / ht - e)
    while l < maxIter and corCaliente:
        
        tArr.append(tArr[l - 1] + ht)

        eqInd = 0   # Indice de la ecuacion, i * cols + j

        # Sistema de ecuaciones para las temperaturas, A x = b
        A = np.zeros((rows * cols, rows * cols))
        b = np.zeros(rows * cols)

        for i in range(0, rows):
            for j in range(0, cols):

                A[eqInd][i * cols + j] = 1 + 2 * k[i][j] * ht * (1 / (hx ** 2) + 1 / (hy ** 2)) 
                b[eqInd] = TArrArrArr[l - 1][i][j]
                                
                if i + 1 < rows:
                    A[eqInd][(i + 1) * cols + j] = - k[i + 1][j] * ht / (hy ** 2)
                else:
                    b[eqInd] += T_BORDE * K_LIQUIDO * ht / (hy ** 2) # Borde superior

                if i - 1 >= 0:
                    A[eqInd][(i - 1) * cols + j] = - k[i - 1][j] * ht / (hy ** 2)                     
                else:
                    b[eqInd] += T_BORDE * K_LIQUIDO * ht / (hy ** 2) # Borde inferior
                
                if j + 1 < cols:                   
                    A[eqInd][i * cols + (j + 1)] = - k[i][j + 1] * ht / (hx ** 2)  
                else:
                    b[eqInd] += T_BORDE * K_LIQUIDO * ht / (hx ** 2) # Borde derecho
                    
                if j - 1 >= 0: 
                    A[eqInd][i * cols + (j - 1)] = - k[i][j - 1] * ht / (hx ** 2) 
                else:
                    b[eqInd] += T_BORDE * K_LIQUIDO * ht / (hx ** 2) # Borde izquierdo

                eqInd += 1

        
        solTl = np.reshape(np.linalg.solve(A, b), (rows, cols))
        TArrArrArr.append(solTl)

        corCaliente = np.any(solTl > TLimInf)
        if not corCaliente:
            tEnfri = t0 + l * ht
        else:
            l += 1


    return tEnfri, tArr, TArrArrArr

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Auxiliares:
# https://www.unitconverters.net/typography/millimeter-to-pixel-x.htm
def mmAPixel(mm):
    return math.floor(mm * 3.7795275591)

# Interpola desde 0 hue (rojo) a 0.62 (azul)
def colorFader(mix): 
    hsvColor = colorsys.rgb_to_hsv(1., 0., 0.)
    return np.asarray(colorsys.hsv_to_rgb(mix, hsvColor[1], hsvColor[2]))

def graficar(TArrArrArr, t, ht):

    #Interpolacion lineal entre naranja y azul
    y = lambda x: - 0.62 / (T0_INTERIOR - T0_LIQUIDO) * (x - T0_LIQUIDO) + 0.62  
    l = math.floor(max(ht, t) / ht) - 1

    _, rows, cols = np.shape(TArrArrArr)
    Trgb = np.zeros((rows, cols, 3))

    for i in range(rows):
       for j in range(cols):
        T = TArrArrArr[l][i][j]
        Trgb[i][j] = colorFader(y(T))

    mplplt.imshow(Trgb)#, aspect = 'auto')
    cantDig = 4
    mplplt.title('t = ' + str(round(t, cantDig)) + 's')
    mplplt.show()

# -------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    imgCor = mplimg.imread(__file__ + '\..\corazon.png')
    t0 = 0.
    tf = 20. * 60.
    hx = 5.
    hy = hx
    ht = 10. 
    imgCorMuest, imgCorDiscret = discretizar(imgCor, hx, hy)
    
    mplplt.imshow(imgCorMuest)#, aspect = 'auto')
    mplplt.show()

    mplplt.imshow(imgCorDiscret)#, aspect = 'auto')
    mplplt.show()

    timer0 = timer()
    tEnfri, tArr, TArrArrArr = diferenciasFinitas(imgCorDiscret, t0, tf, ht, hx, hy) 
    timerF = timer()
    
    print("Tiempo de cálculo: " + str(round(timerF - timer0, 4)) + "s")
    print("Tiempo de enfriamiento para llegar a por debajo de " + str(T_ENFRIAMIENTO) + "° C: " + str(tEnfri) + "s")
    
    cantGraficos = 1 + 4
    for t in np.linspace(t0, tEnfri, cantGraficos):
        graficar(TArrArrArr, t, ht)
    

    
    

        

   


    