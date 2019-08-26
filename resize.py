import glob
from PIL import Image

#lista = ['edible', 'inedible']
from PIL import Image
for i in range (len(lista)):
    aux = lista[i]
    print(aux)
    i = i + 1
    string = ('/data/train/edible/*')
    files = glob.glob(string, recursive=False)
    for i in range (len(files)):
        im = Image.open(files[i])
        width, height = im.size	
        im = im.resize((128,96))
        im.save(files[i])
        print (files[i])