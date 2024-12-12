import jetFunctions as j
import time

measure = []
try:
    number_of_measures = 910
    j.initSpiAdc()
    j.initStepMotorGpio()
    for i in range(number_of_measures):
        for i_2 in range(3):
            time.sleep(0.01)
            measure.append(j.getAdc())
        j.stepForward(1)
        print(i)

finally:
    j.stepBackward(int(number_of_measures))
    j.deinitSpiAdc()
    j.deinitStepMotorGpio()
    j.jsave("data100.txt", measure)