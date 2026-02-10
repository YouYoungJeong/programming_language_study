class ElecProduct:
    volume = 0
    
    def valumeControl(self, volum):
        # print(f"{volum} ElecProduct의 valumeControl")
        pass
    
    def sound(self):
        print('ElecProduct의 sound')

class ElecTv(ElecProduct):
    def valumeControl(self, volum):
        self.volume = volum
        print('불금 와우')
        print("ElecTv의 valumeControl")

class ElecRadio(ElecProduct):
    def valumeControl(self, volum):
        sori = volum
        print("ElecRadio의 valumeControl")


product = ElecProduct()
tv = ElecTv()
product  = tv
product.valumeControl(5)

radio = ElecRadio()
product = radio
product.valumeControl(3)

print('----------------------')
li = [ElecTv(), ElecRadio()]

for a in li:
    a.valumeControl(2)