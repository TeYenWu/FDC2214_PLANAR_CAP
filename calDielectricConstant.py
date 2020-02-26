import math

def calConstant():
    dielectConstant = [110, 250, 175, 600, 260, 100, 120, 180, 200]
    lossTangent = [2.727, 3.2, 4.85, 2.16, 2.307, 4, 5, 3.88, 3]
    i = 0

    for item in dielectConstant:
        k = -lossTangent[i]*(2/math.pi)
        dielectConstant[i] = dielectConstant[i]*math.pow(10, k)
        print("material[" + str(i) + "]'s dielectric constant at 1MHz: " + str(dielectConstant[i]))
        i += 1


if __name__ == '__main__':
    calConstant()