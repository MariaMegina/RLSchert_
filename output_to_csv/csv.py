f1 = open('output_75_different_symbol.txt', 'r')
f2 = open('output_75_different_symbol.csv', 'w')
for i in  f1:
    s = ""
    for c in i:
        if (c==' '):
            c = ';'
        elif (c=='['):
            c = ''
        elif (c==']'):
            c = ''
        elif (c=='.'):
            c = ''
        s += c
    f2.write(s)

f1.close()
f2.close()
        