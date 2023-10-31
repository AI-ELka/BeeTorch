import mysql.connector as mysql
import numpy as np
import matplotlib.pyplot as plt

authF = open('conf/sql.txt', 'r')
auth = authF.readlines()

db = mysql.connect(
            host = auth[0],
            user = auth[1],
            passwd = auth[2].replace("\n",""),
            database = auth[3]
)

cursor = db.cursor()

modelName="Polynomial_Regression"

def show(detail=False):
    if(detail):
        show_detail()
        return

    cursor.execute("SELECT model_id,dimension,max(accuracy) from logs,models where logs.model_id=models.id and poison=0 GROUP BY model_id")

    result=cursor.fetchall()
    result = np.array(result)
    dataX = result[:,1]
    dataY = result[:,2]
    plt.scatter(dataX,dataY)
    plt.show()

treshhold = [2000,4000]

def show_detail():
    cursor.execute(f"SELECT table1.model_id,dimension,table1.accuracy,max.max_epochs-min(table1.epochs) as dif_epochs FROM logs as table1,(SELECT model_id,poison,poison_rate,max(accuracy) as accuracy,max(epochs) as max_epochs FROM logs GROUP BY model_id,poison,poison_rate) as max, models WHERE models.name='{modelName}' and table1.accuracy=max.accuracy and table1.model_id=max.model_id and table1.poison=max.poison and table1.poison_rate=max.poison_rate and table1.poison=0 and table1.model_id=models.id GROUP BY table1.model_id,table1.accuracy,table1.poison,table1.poison_rate ORDER BY table1.model_id;")

    result=cursor.fetchall()
    #print(result)
    plt.scatter([result[i][1]/7840 for i in range(len(result)) if result[i][3]>=treshhold[1]],[result[i][2] for i in range(len(result)) if result[i][3]>=treshhold[1]],color="green")
    plt.scatter([result[i][1]/7840 for i in range(len(result)) if result[i][3]<treshhold[1] and result[i][3]>=treshhold[0]],[result[i][2] for i in range(len(result)) if result[i][3]<treshhold[1] and result[i][3]>=treshhold[0]],color="orange")
    plt.scatter([result[i][1]/7840 for i in range(len(result)) if result[i][3]<treshhold[0]],[result[i][2] for i in range(len(result)) if result[i][3]<treshhold[0]],color="red")
    plt.show()


def evolution(model_id):
    cursor.execute(f"SELECT epochs,accuracy FROM logs WHERE model_id={model_id} ORDER BY epochs")

    result=cursor.fetchall()
    result=np.array(result)
    plt.plot(result[:,0],result[:,1])
    plt.title(str(model_id))
    plt.show()

def loss(model_id):
    cursor.execute(f"SELECT epochs,loss FROM logs WHERE model_id={model_id} ORDER BY epochs")

    result=cursor.fetchall()
    result=np.array(result)
    plt.plot(result[:,0],result[:,1])
    plt.title(str(model_id))
    plt.show()