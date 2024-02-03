import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import math
import os

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Inciso 1: integracion numerica. 
# Se supone que xArrArr = np.array([[x1(t0), x1(t0 + h), x1(t0 + 2 * h)], ... , [xn(t0), xn(t0 + h), xn(t0 + 2 * h)]]) o xArr = np.array([[x(t0), x(t0 + h), x(t0 + 2 * h)])
# (matriz o arreglo cuyas filas de 3 elementos son evaluaciones equiespaciadas de la/s funcion/es)
def trapecios3(a, b, t0, h, xArrArr):
    # Antiderivadas de las rectas que aproximan a x(t)
    X1 = lambda t: xArrArr[0] * t + (xArrArr[1] - xArrArr[0]) / h * (t ** 2 / 2 - t0 * t)
    X2 = lambda t: xArrArr[1] * t + (xArrArr[2] - xArrArr[1]) / h * (t ** 2 / 2 - (t0 + h) * t)

    # Integracion, TFC
    return (X1(t0 + h) - X1(a)) + (X2(b) - X2(t0 + h))

def simpson3(a, b, t0, h, xArrArr):
    # Antiderivada de la parabola que aproxima a x(t)
    d1 = (xArrArr[1] - xArrArr[0]) / h
    d2 = (xArrArr[2] - xArrArr[1]) / h
    c1 = (d2 - d1) / (6 * h)
    c2 = d1 / 2 + (d1 - d2) * (2 * t0 + h) / (4 * h)
    c3 = xArrArr[0] + t0 * ((t0 + h) * (d2 - d1) / (2 * h) - d1)
    
    X = lambda t: c1 * t**3 + c2 * t**2 + c3 * t

    # Integracion, TFC
    return X(b) - X(a)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
e = 1e-4 # Epsilon para el limite superior de iteracion en RK4 y Milne

# Inciso 2 / 3: Milne vectorial, o predictor-corrector-supracorrector vectorial
# Resuelve sistemas de EDOs, cada ecuacion de la forma xi'(x) = fi(t, xArr), t indepediente y escalar, xArr = x1, x2, ..., xn
def MilneVect(t0, tf, x0Arr, h, fVect, correc = 2):

    maxIter = math.ceil((tf - t0) / h - e)
    iterRK4 = 3

    # Hacer 3 pasos de RK4 para tener datos para Milne
    tArr, xArrArr, dxArrArr = RK4Vect(t0, t0 + iterRK4 * h, x0Arr, h, fVect) 

    # Expandir arreglos
    tArr = np.append(tArr, np.zeros(maxIter - iterRK4))
    xArrArr  = np.append(xArrArr,  np.zeros((maxIter - iterRK4, len(x0Arr))), axis = 0)
    dxArrArr = np.append(dxArrArr, np.zeros((maxIter - iterRK4, len(x0Arr))), axis = 0)

    # Ciclo principal
    for i in range(3, maxIter):
        # Predecir
        tArr[i + 1] = tArr[i] + h
        xi1Arr = xArrArr[i - 3] + simpson3(tArr[i - 3], tArr[i + 1], tArr[i - 2], h, dxArrArr[i - 2 : i + 1])
        dxArrArr[i + 1] = fVectEval(fVect, tArr[i + 1], xi1Arr)

        # Corregir, supracorregir, ...
        for j in range(correc):
            xi1ArrOld = xi1Arr
            xi1Arr = xArrArr[i - 1] + simpson3(tArr[i - 1], tArr[i + 1], tArr[i - 1], h, dxArrArr[i - 1 : i + 2])
            # Sumar estimador del error
            xi1Arr = xi1Arr + estErrorMilne(xi1Arr, xi1ArrOld)
            
            dxArrArr[i + 1] = fVectEval(fVect, tArr[i + 1], xi1Arr)
      
        xArrArr[i + 1] = xi1Arr

    return tArr, xArrArr, dxArrArr

# Richardson para Milne
def estErrorMilne(xC, xP):
    return (xC - xP) / -29

# Milne escalar, o predictor-corrector-supracorrector escalar
# Resuelve EDOs de la forma x'(t) = f(t, x), t independiente
def MilneScal(t0, tf, x0, h, f, correc = 2):
    x0Arr = np.array([x0])
    fVect = [lambda t, xArr: f(t, xArr[0])]
    tArr, xArrArr, dxArrArr = MilneVect(t0, tf, x0Arr, h, fVect, correc)

    return tArr, xArrArr[:, 0], dxArrArr[:, 0]

# --------------------------------

# Runge-Kutta 4 vectorial
# Resuelve sistemas de EDOs, cada ecuacion de la forma xi'(x) = fi(t, xArr), t indepediente y escalar, xArr = x1, x2, ..., xn
def RK4Vect(t0, tf, x0Arr, h, fVect):

    maxIter = math.ceil((tf - t0) / h - e)

    # Crear e inicializar arreglos con condiciones iniciales
    tArr = np.zeros(maxIter + 1)
    tArr[0] = t0

    xArrArr = np.zeros((maxIter + 1, len(x0Arr)))
    xArrArr[0] = x0Arr
    
    dxArrArr = np.zeros((maxIter + 1, len(x0Arr)))
    dxArrArr[0] = fVectEval(fVect, t0, x0Arr)

    # Ciclo principal
    for i in range(0, maxIter):
        tArr[i + 1] = tArr[i] + h

        # Aproximar con paso h y h/2
        xi1ArrH = RK4VectEval(tArr[i], xArrArr[i], fVect, h)
        xi1ArrH2 = RK4VectEval(tArr[i], xArrArr[i], fVect, h/2)
        xi1ArrH2 = RK4VectEval(tArr[i] + h/2, xi1ArrH2, fVect, h/2)
        
        # Sumar estimador del error
        xi1ArrH2 = xi1ArrH2 + estErrorRK4(xi1ArrH2, xi1ArrH)

        xArrArr[i + 1] = xi1ArrH2
        dxArrArr[i + 1] = fVectEval(fVect, tArr[i + 1], xi1ArrH2)
        
    return tArr, xArrArr, dxArrArr
    
def RK4VectEval(t, xArr, fVect, h):
    
    k1 = fVectEval(fVect, t, xArr)
    k2 = fVectEval(fVect, t + h/2, xArr + h/2 * k1)
    k3 = fVectEval(fVect, t + h/2, xArr + h/2 * k2)
    k4 = fVectEval(fVect, t + h, xArr + h * k3)
   
    return xArr + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4) 


# Richardson para RK4
def estErrorRK4(xH2, xH):
    return (xH2 - xH) / 15

# Runge-Kutta 4 escalar
# Resuelve EDOs de la forma x'(t) = f(t, x), t independiente
def RK4Scal(t0, tf, x0, h, f):
    x0Arr = np.array([x0])
    fVect = [lambda t, xArr: f(t, xArr[0])]
    tArr, xArrArr, dxArrArr = RK4Vect(t0, tf, x0Arr, h, fVect)

    return tArr, xArrArr[:, 0], dxArrArr[:, 0]

# --------------------------------

def fVectEval(fVect, t, xArr):
    res = np.zeros(len(fVect))
    for j in range(len(fVect)):
        res[j] = fVect[j](t, xArr)
    
    return res

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Funciones auxiliares
def plot(indVarArr, depVarTupList, indVarLabel, title):
    
    for i in range(len(depVarTupList)):
        depVarArr, depVarColor, depVarLabel = depVarTupList[i]
        plt.plot(indVarArr, depVarArr, color = depVarColor, label = depVarLabel)
    
    plt.xlabel(indVarLabel)
    plt.title(title)
    plt.legend()
    plt.show()

def pause():
    input("\nPress Enter to continue...")
    os.system('cls')

def inciso1():
    # Pequenio ejemplo. Cambiar parametros a placer
    t = sp.Symbol('t')
    x = sp.sin(t)
    t0 = -np.pi/2
    h = 1.
    #xArr = np.array(list(map(sp.lambdify(t, x), np.arange(x0, x0 + 2 * h + e, h))))
    xArr = np.array([x.subs(t, t0), x.subs(t, t0 + h), x.subs(t, t0 + 2 * h)])

 
    print("x(t) = " + str(x) + "; t0 = " + str(t0) + "; h = " + str(h))

    a = t0 + h
    b = t0 + 2.5 * h
    print("\nFórmula semiabierta")
    print("I_exacta    = " + str(sp.integrate(x, (t, a, b))))
    print("I_trapecios = " + str(trapecios3(a, b, t0, h, xArr)))
    print("I_simpson   = " + str(simpson3(a, b, t0, h, xArr)))

    a = t0 + 0.5 * h
    b = t0 + 1.5 * h
    print("\nFórmula supracerrada")
    print("I_exacta    = " + str(sp.integrate(x, (t, a, b))))
    print("I_trapecios = " + str(trapecios3(a, b, t0, h, xArr)))
    print("I_simpson   = " + str(simpson3(a, b, t0, h, xArr)))


    pause()

def inciso2():
    # Ejemplo de aplicacion, visto en clase
    t0 = 0.
    tf = 20.
    x0 = 1
    h = 0.2
    f = lambda t, x: t + 1/5 * np.cos(t + x)

    tArr, xArr, dxArr = MilneScal(t0, tf, x0, h, f)

    fixDec = "{:8.8f}"
    print("t\t\tx(t)\t\tx'(t)\n")
    for i in range(len(xArr)):
        print(str(fixDec.format(tArr[i])) + '\t' + str(fixDec.format(xArr[i])) + '\t' + str(fixDec.format(dxArr[i])))
   
    # x y dx/dt
    plot(tArr, [(xArr, 'green', 'x'), (dxArr, 'limegreen', 'dx/dt')], 't', 'Ejemplo de aplicación: x\'(t) = t + 1/5 * cos(t + x)')


    pause()

def inciso3():
    t0 = 0.
    tf = 20.
    h = 0.01
    # xArr[0] = V; xArr[1] = W
    V0 = 0.
    W0 = 0.
    x0Arr = np.array([V0, W0])
    Z = 0
    fVect = []
    fVect.append(lambda t, xArr: 3 * (xArr[0] + xArr[1] - xArr[0]**3 / 3 + Z))
    fVect.append(lambda t, xArr: -1/3 * (xArr[1] - 0.7 + 0.8 * xArr[0]))
    
    tArr, xArrArr, dxArrArr = MilneVect(t0, tf, x0Arr, h, fVect)

    # V y dV/dt
    plot(tArr, [(xArrArr[:, 0], 'red', 'V'), (dxArrArr[:, 0], 'sandybrown', 'dV/dt')], 't', 'Potencial de acción, V')
    # W y dW/dt
    plot(tArr, [(xArrArr[:, 1], 'blue', 'W'), (dxArrArr[:, 1], 'cornflowerblue', 'dW/dt')], 't', 'Recuperación, W')
    # V y W
    plot(tArr, [(xArrArr[:, 0], 'red', 'V'), (xArrArr[:, 1], 'blue', 'W')], 't', 'Diagrama de estado V - W')

# -----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    inciso1()
    inciso2()
    inciso3()




    


