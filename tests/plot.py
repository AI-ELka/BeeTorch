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

def show():
    cursor.execute("SELECT model_id,dimension,max(accuracy) from logs,models where logs.model_id=models.id and poison=0 GROUP BY model_id")

    result=cursor.fetchall()
    result = np.array(result)
    dataX = result[:,1]
    dataY = result[:,2]
    plt.scatter(dataX,dataY)
    plt.show()