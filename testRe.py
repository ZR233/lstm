import re
a = '123.jpg'
pattern = re.compile(r'(.jpg)')
b = re.findall(r'jpg$',a)
if b:
    print(b)
