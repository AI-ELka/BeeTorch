from pushbullet import Pushbullet
import time
from beetorch import Saver,Poison


def default_text(name,dimension,dataset,epochs,accuracy,loss,poison,poisonRate):
    poison_txt = Poison.toString(poison)
    if poison==Poison.NO_POISONING:
        return ["["+dataset+"] Log for "+name+" with "+str(dimension)+" dimension :","Epoch : "+str(epochs)+f" | Loss : {loss:.5f}"+f" - Accuracy : {accuracy}"]
    return ["["+dataset+"] Log for "+name+" with "+str(dimension)+f" dimension and with a {int(poisonRate*100)}% {poison_txt} :","Epoch : "+str(epochs)+f" | Loss : {loss:.5f}"+f" - Accuracy : {accuracy}"]
    

class Pushbullet_saver(Saver):
    def __init__(self,access_token=True,minTime=60*3):
        self.time = time.time()
        self.minTime=minTime
        self.texer = default_text
        self.name=""
        self.dimension=0
        self.dataset=""
        if access_token==True:
            authF = open('conf/pushbullet.txt', 'r')
            access_token = authF.readline().rstrip()
        self.access_token=access_token
        self.working=True
        try:
            self.api = Pushbullet(access_token)
            self.channel=self.api.channels[0]
        except:
            print("Couldn't load PushBullet :(")
            self.working=False
        super().__init__()
    
    
    def save_log(self,epochs,accuracy,loss):
        if (not self.initiallized) or (not self.working):
            return
        if self.minTime>time.time()-self.time:
            return
        self.time=time.time()
        try:
            msg=default_text(self.name,self.dimension,self.dataset,epochs,accuracy,loss,self.poison,self.poisonRate)
            self.channel.push_note(msg[0],msg[1])
        except:
            self.api = Pushbullet(self.access_token)
            self.channel=self.api.channels[0]

    