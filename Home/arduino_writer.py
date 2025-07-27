import serial
import time


arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(10) 

angle = 90
arduino.write(f'{angle}\n'.encode())

arduino.close()
