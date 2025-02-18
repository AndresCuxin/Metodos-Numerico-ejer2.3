import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Definir la función original
def f(x):
    return x**3 - 6*x**2 + 11*x - 6

# Función para la interpolación de Lagrange
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

# Método de Bisección para encontrar la raíz
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

# Seleccionar tres puntos adecuados en [1, 3]
x_points = np.array([1, 2, 3])
y_points = f(x_points)

# Construcción del polinomio interpolante
x_vals = np.linspace(1, 3, 100)
y_interp = np.array([lagrange_interpolation(x, x_points, y_points) for x in x_vals])

# Encontrar raíz del polinomio interpolante usando bisección
root, iter_data = bisect(lambda x: lagrange_interpolation(x, x_points, y_points), 1, 3)

# Crear tabla de iteraciones
df_iter = pd.DataFrame(iter_data, columns=["Iteración", "a", "b", "c", "f(c)", "Error"])
print("\nTabla de iteraciones:")
print(df_iter)

# Gráfica
plt.figure(figsize=(8, 6))
plt.plot(x_vals, f(x_vals), label="f(x) = x^3 - 6x^2 + 11x - 6", linestyle='dashed', color='blue')
plt.plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(root, color='green', linestyle='dotted', label=f"Raíz aproximada: {root:.4f}")
plt.scatter(x_points, y_points, color='black', label="Puntos de interpolación")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Interpolación y búsqueda de raíces")
plt.legend()
plt.grid(True)
plt.savefig("grafica_interpolacion.png")  # Guarda la gráfica de interpolación
plt.show()

# Imprimir la raíz y los errores
print(f"Raíz aproximada usando interpolación: {root:.4f}")

