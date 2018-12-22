def printIntersection(arr1, arr2): 
    temp = []
    for num in arr1:
        first = 0
        last = len(arr2) - 1
        while first <= last:
            midpoint = (first + last)//2
            if arr2[midpoint] == num:
                temp.append(num)
                arr2.remove(arr2[midpoint])
                break
            else:
                if num < arr2[midpoint]:
                    last = midpoint -1
                else:
                    first = midpoint +1
    print(temp)
arr1 = [1,2,3,3,4,4,5,6,7,8,9,9]
arr2 = [1,1,2,3,4,4,5,6,6]
# arr1 = range(1000)
# arr2 = range(2000)
from datetime import datetime
s = datetime.now()
printIntersection(arr1, arr2)
e = datetime.now()
print(e-s)