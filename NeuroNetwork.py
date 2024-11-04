import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, input_shape=(3,), activation='relu'),
    Dense(32, activation='relu'),                     
    Dense(2, activation='softmax')                   
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


X_train = np.random.rand(100, 3)  
y_train = np.random.randint(2, size=100)  

X_test = np.random.rand(20, 3) 
y_test = np.random.randint(2, size=20) 


model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


predictions = model.predict(X_test)
print("Predictions:", predictions)
