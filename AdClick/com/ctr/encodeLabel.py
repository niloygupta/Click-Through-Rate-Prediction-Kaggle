'''
Created on 31-Dec-2014

@author: niloygupta
'''
import psycopg2
import sys
import pickle
import numpy as np
from sklearn import preprocessing

db_conn = None;



def main():
    global db_conn

    try:
        db_conn = psycopg2.connect(database='adclick', user='postgres', port='5433')
        cur = db_conn.cursor()
        labels = []
        categories = ['site_id','site_domain','site_category','app_id','app_domain','app_category','device_model']
        table_name = ['siteId','siteDomain','siteCategory','appId','appDomain','appCategory','deviceModel']
        table_id = 0
        for category in categories:
            cur.execute("select %s from %s" %(category,table_name[table_id]))
            queryResult = cur.fetchall()
            dataChunk  = np.array(queryResult)
            le = preprocessing.LabelEncoder()
            le.fit(dataChunk[:,0])
            labels.append(le)
            print "Done %s" %(table_name[table_id])
            table_id = table_id + 1
        db_conn.close()
        pickle.dump( labels, open( "label.p", "wb" ) )

    except:
        print "Unexpected error:", sys.exc_info()[0]    
        if (db_conn):
            db_conn.close()
            
        raise                    

if __name__ == '__main__':
    main()