# http://www.sigcon.com/Pubs/news/4_5.htm
import math

def calConstant1():
    dielectConstant = [110, 250, 175, 600, 260, 100, 120, 180, 200]
    lossTangent = [2.727, 3.2, 4.85, 2.16, 2.307, 4, 5, 3.88, 3]
    i = 0

    for item in dielectConstant:
        k = -math.atan(lossTangent[i])*(2/math.pi)
        print(math.atan(lossTangent[i]))
        dielectConstant[i] = dielectConstant[i]*math.pow(100, k)
        print("material[" + str(i) + "]'s dielectric constant at 1MHz: " + str(dielectConstant[i]))
        i += 1

def calConstant2():
    objectName = []
    absolutePermitti = 8.854187817*math.pow(10, -12)
    vaccumCapaci = "测定"
    objectCapaci = ["测定"]
    objectConstant = []
    i = 0

    for item in objectName:
        objectConstant[i] = absolutePermitti*(objectCapaci[i]/vaccumCapaci[i])
        print("object["+str(i)+"]: "+objectName[i]+"'s dielectric constant = "+str(objectConstant[i]))
        i += 1

if __name__ == '__main__':
    calConstant2()