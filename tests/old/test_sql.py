from beetorch.sql import SQL_saver
import asyncio


sql = SQL_saver()

sql.init("MNIST_REGRESSION",7840,"MNIST")
print("Launched init")

