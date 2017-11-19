import sqlite3

import ijson
import marshal
import matplotlib.pyplot as plt
import numpy as np


class DataSample:
    def __init__(self, id, band1, band2, angle):
        self.id = id
        self.band1 = band1
        self.band2 = band2
        self.angle = angle


def create_sqlite_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return None


def create_sql_table(conn):
    try:
        sqlCreateStatement = "CREATE TABLE IF NOT EXISTS samples ( id TEXT PRIMARY KEY, band1 TEXT NOT NULL, band2 TEXT NOT NULL, angle TEXT NOT NULL)"
        c = conn.cursor()
        c.execute(sqlCreateStatement)
    except sqlite3.Error as e:
        print(e)


def insert_data_sample(conn, task):
    sql = ''' INSERT INTO samples(id,band1,band2,angle)
              VALUES(?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, task)
    conn.commit()
    return cur.lastrowid


def parseJsonFileToSQLITE(file, conn):
    for prefix, event, value in ijson.parse(file):
        if (event) == ("start_map"):
            id = ""
            band1 = []
            band2 = []
            angle = 0
        if prefix == "item.id":
            id = value
        if prefix == "item.band_2.item":
            band2.append(str(value))
        if prefix == "item.band_1.item":
            band1.append(str(value))
        if prefix == "item.inc_angle":
            angle = value
        if event == "end_map":
            insert_data_sample(conn, (id, marshal.dumps(band1), marshal.dumps(band2), str(angle),))


def startUp(filename, db_file):
    conn = create_sqlite_connection(db_file)
    # create_table(conn)
    # parseJsonFile(open(filename),conn)
    drawTestSample(conn)


def getFirstSamplesFromDB(conn):
    try:
        sqlGet = "SELECT * from samples LIMIT 100"
        cur = conn.cursor()
        cur.execute(sqlGet)
        rows = cur.fetchall()
        dataSampleList = []
        for row in rows:
            sampleNew = DataSample(row[0], marshal.loads(row[1]), marshal.loads(row[2]), row[3])
            dataSampleList.append(sampleNew)
        return dataSampleList

    except sqlite3.Error as e:
        print(e)
        return None


def drawTestSample(conn):
    sampleList = getFirstSamplesFromDB(conn)
    for i in sampleList:
        hd = np.asarray(i.band1, dtype='float').reshape((75, 75))
        vd = np.asarray(i.band2, dtype='float').reshape((75, 75))
        full = np.concatenate((hd, vd))
        plt.imshow(full)
        plt.show()

    return None


if __name__ == '__main__':
    db_file = "db.db"
    filename = "data/test.json"

    startUp(filename, db_file)
