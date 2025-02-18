import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Función original
def f(x):
    return np.exp(-x) - x

# Interpolación de Lagrange
def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

# Método de Bisección con almacenamiento de iteraciones
def bisect(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:
        raise ValueError("El intervalo no contiene una raíz")
    
    iter_data = []
    for i in range(max_iter):
        c = (a + b) / 2
        error = abs(func(c))
        iter_data.append([i + 1, a, b, c, func(c), error])
        
        if error < tol or (b - a) / 2 < tol:
            return c, iter_data
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2, iter_data  # Mejor estimación de la raíz

# Selección de al menos cuatro puntos en [0,1]
x_points = np.linspace(0, 1, 4)
y_points = f(x_points)

# Construcción del polinomio interpolante
x_vals = np.linspace(0, 1, 100)
y_interp = np.array([lagrange_interpolation(x, x_points, y_points) for x in x_vals])

# Encontrar raíz del polinomio interpolante usando bisección
root, iter_data = bisect(lambda x: lagrange_interpolation(x, x_points, y_points), 0, 1)

# Crear tabla de iteraciones
df_iter = pd.DataFrame(iter_data, columns=["Iteración", "a", "b", "c", "f(c)", "Error"])
print("\nTabla de iteraciones:")
print(df_iter)

# Gráfica
plt.figure(figsize=(8, 6))
plt.plot(x_vals, f(x_vals), label="f(x) = e^(-x) - x", linestyle='dashed', color='blue')
plt.plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(root, color='green', linestyle='dotted', label=f"Raíz aproximada: {root:.4f}")
plt.scatter(x_points, y_points, color='black', label="Puntos de interpolación")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Interpolación y búsqueda de raíces")
plt.legend()
plt.grid(True)
plt.savefig("grafica_errores.png")  # Guarda la gráfica de errores
plt.show()

# Imprimir la raíz y los errores
print(f"Raíz aproximada usando interpolación: {root:.4f}")
