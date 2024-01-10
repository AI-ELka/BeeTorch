import mysql.connector as mysql
import numpy as np
import matplotlib.pyplot as plt

authF = open('conf/sql2.txt', 'r')
auth = authF.readlines()

db = mysql.connect(
            host = auth[0],
            user = auth[1],
            passwd = auth[2].replace("\n",""),
            database = auth[3]
)

cursor = db.cursor()

def reload():
    db = mysql.connect(
            host = auth[0],
            user = auth[1],
            passwd = auth[2].replace("\n",""),
            database = auth[3]
    )

    cursor = db.cursor()

modelName="Polynomial_Regression"

def show(poison=0,poisonRate=0,detail=False):
    if(detail):
        show_detail(poison,poisonRate)
        return

    cursor.execute(f"SELECT model_id,dimension,max(accuracy) from logs,models where models.name='Polynomial_Regression_Float32' and logs.model_id=models.id and poison={poison} and ABS(poison_rate-{poisonRate})<0.00001 GROUP BY model_id")

    result=cursor.fetchall()
    result = np.array(result)
    try:
        dataX = result[:,1]
        dataY = result[:,2]
        plt.scatter(dataX,dataY)
        plt.show()
    except:
        print(result)

treshhold = [2000,6000]
trie = 'and try=1'
poisons = ((0,0),(1,.1),(1,.2),(1,.3),(1,.4))
colors = ('blue','red','green','purple','orange')
def label_construct(poison):
    string=""
    if poison[0]==0:
        string += "No poison"
    elif poison[0]==1:
        string+="Label Flipping at "+str(int(poison[1]*100))+"%"
    return string
def all(float64=False):
    for i,poison in enumerate(poisons):    
        cursor.execute(f"SELECT model_id,dimension,max(accuracy) from logs,models where models.name='Polynomial_Regression_Float32' and logs.model_id=models.id {trie} and poison={poison[0]} and ABS(poison_rate-{poison[1]})<0.00001 GROUP BY model_id")
        result=cursor.fetchall()
        result = np.array(result)
        plt.scatter(result[:,1],result[:,2],color=colors[i],label=label_construct(poison))
        if float64:
            try:
                cursor.execute(f"SELECT model_id,dimension,max(accuracy) from logs,models where models.name='Polynomial_Regression' and logs.model_id=models.id and poison={poison[0]} and ABS(poison_rate-{poison[1]})<0.00001 GROUP BY model_id")
                result=cursor.fetchall()
                result = np.array(result)
                plt.scatter(result[:,1],result[:,2],color=colors[i],marker=',')
            except:
                1==1
    plt.legend()
    plt.show()

    cursor.execute(f"SELECT model_id,dimension,max(accuracy) from logs,models where  models.name='Polynomial_Regression_Float32' and logs.model_id=models.id and poison={1} and ABS(poison_rate-{0.1})<0.00001 GROUP BY model_id")

    result2=cursor.fetchall()
    
    result2 = np.array(result2)



def show_detail(poison=0,poisonRate=0):
    cursor.execute(f"SELECT table1.model_id,dimension,table1.accuracy,max.max_epochs-min(table1.epochs) as dif_epochs FROM logs as table1,(SELECT model_id,poison,poison_rate,max(accuracy) as accuracy,max(epochs) as max_epochs FROM logs GROUP BY model_id,poison,poison_rate) as max, models WHERE models.name='{modelName}' and table1.accuracy=max.accuracy and table1.model_id=max.model_id and table1.poison=max.poison and table1.poison_rate=max.poison_rate and table1.poison={poison} and ABS(table1.poison_rate-{poisonRate})<0.00001 and table1.model_id=models.id GROUP BY table1.model_id,table1.accuracy,table1.poison,table1.poison_rate ORDER BY table1.model_id;")

    result=cursor.fetchall()
    #print(result)
    plt.scatter([result[i][1]/7840 for i in range(len(result)) if result[i][3]>=treshhold[1]],[result[i][2] for i in range(len(result)) if result[i][3]>=treshhold[1]],color="green")
    plt.scatter([result[i][1]/7840 for i in range(len(result)) if result[i][3]<treshhold[1] and result[i][3]>=treshhold[0]],[result[i][2] for i in range(len(result)) if result[i][3]<treshhold[1] and result[i][3]>=treshhold[0]],color="orange")
    plt.scatter([result[i][1]/7840 for i in range(len(result)) if result[i][3]<treshhold[0]],[result[i][2] for i in range(len(result)) if result[i][3]<treshhold[0]],color="red")
    plt.show()


def evolution(model_id,poison=0,poisonRate=0,try_num=0):
    cursor.execute(f"SELECT epochs,accuracy FROM logs WHERE model_id={model_id} and poison={poison} and ABS(poison_rate-{poisonRate})<0.00001 and try={try_num} ORDER BY epochs")

    result=cursor.fetchall()
    result=np.array(result)
    plt.plot(result[:,0],result[:,1])
    plt.title(str(model_id))
    plt.show()

def loss(model_id,poison=0,poisonRate=0,try_num=0):
    cursor.execute(f"SELECT epochs,loss FROM logs WHERE and model_id={model_id} and poison={poison} and ABS(poison_rate-{poisonRate})<0.00001 and try={try_num} ORDER BY epochs")

    result=cursor.fetchall()
    result=np.array(result)
    plt.plot(result[:,0],result[:,1])
    plt.title(str(model_id))
    plt.show()




