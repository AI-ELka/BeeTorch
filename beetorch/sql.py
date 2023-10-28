import asyncio
import mysql.connector as mysql


class SQL_saver:
    def __init__(self,host=True,username=True,passw=True,database=True):
        self.host=host
        self.username=username
        self.passw=passw
        self.database=database
        if host==True or username==True or passw==True or database==True:
            authF = open('conf/sql.txt', 'r')
            auth = authF.readlines()
            if host==True:
                self.host=auth[0]
            if username==True:
                self.username=auth[1]
            if passw==True:
                self.passw=auth[2]
            if database==True:
                self.database=auth[3]

        self.initiallized=False

        
        return
    
    
    def init(self,name,dimension,dataset):
        self.db = mysql.connect(
            host = self.host,
            user = self.username,
            passwd = self.passw.replace("\n",""),
            database = self.database
        )
        self.cursor = self.db.cursor()

        self.cursor.execute("SELECT id FROM models WHERE name=%s and dimension=%s",(name,dimension))
        result=self.cursor.fetchall()
        print(result)
        if(len(result)==0):
            self.cursor.execute("INSERT INTO models (name,dimension,dataset) VALUES (%s,%s,%s)",(name,dimension,dataset))
            self.db.commit()
            self.model_id = self.cursor.lastrowid
            return True
        else:
            self.model_id=result[0][0]
            return True
    
    
    async def save_log_async(self,epochs,accuracy,loss):
        if not self.initiallized:
            return False
        sql= "INSERT INTO logs (model_id,epochs,accuracy,loss) VALUES (%s,%s,%s,%s)"
        val= (self.model_id,epochs,accuracy,loss)
        self.cursor.execute(sql,val)
        self.db.commit()
        return True
    
    def save_log(self,epochs,accuracy,loss):
        asyncio.run(self.save_log_async(epochs,accuracy,loss))

