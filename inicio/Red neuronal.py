import tensorflow  as tf
import numpy as np

celsius =np.array([-40, -10, 0, 8, 15, 22, 38, 48, 58,68,78,88,98,108], dtype=float)
Fahrenheit= np.array([-40,-14, 32, 46, 59, 72, 100, 118,136,154,172,190,208,226], dtype=float)

capa = tf.keras.layers.Dense(units=1,  input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial =  modelo.fit(celsius,Fahrenheit, epochs=500, verbose=False)
print("Modelo entrenado!")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history['loss'])
plt.show()

print("Hagamos una predicción!")
resultado = modelo.predict(np.array([100.0]))
print("El resultado es "+ str(resultado)+ "fahrenheit")