print("Nhap day so cua ban")
number = input()
test = []
while True:
    try:
        test = [int(char) for char in number]
        break
    except ValueError:
        print("day so ban nhap khong dung vui vong nhap lai")
        number = input()
        continue
print(test)
count = [0]*(len(test))
count[0] = 1
if (test[0]*10 + test[1]) < 27:
    count[1] = 2
else:
    count[1] = 1
for i in range(2, len(test)):
    A = 0
    if (test[i-1]*10 + test[i])<= 27:
        A =  count[i-2]
    count[i] = count[i-1] + A
print(count[len(count)-1])
