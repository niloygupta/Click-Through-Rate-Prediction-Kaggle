'''
Created on 31-Dec-2014

@author: niloygupta
'''
import psycopg2
import sys
import numpy as np
import scipy
from sklearn.ensemble import RandomForestClassifier

db_conn = None;


def main():
    global db_conn

    try:
        db_conn = psycopg2.connect(database='adclick', user='postgres', port='5433')
        cur = db_conn.cursor()
        categories = ['site_id']#,'site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model']
        table_name = ['siteId']#,'siteDomain','siteCategory','appId','appDomain','appCategory','deviceId','deviceIP','deviceModel']
        table_id = 0
        for category in categories:
            cur.execute("select id,%s from %s" %(category,table_name[table_id]))
            queryResult = cur.fetchall()
            for result in queryResult:
                cur.execute("update adclickdata set %s = '%s' where %s='%s'" %(category,result[1],category,result[0]))
                db_conn.commit()
                print "Completed ID: %s" %(result[0])
            print "Done %s" %(table_name[table_id])
            table_id = table_id + 1
        db_conn.close()
    except:
        print "Unexpected error:", sys.exc_info()[0]    
        if (db_conn):
            db_conn.close()
            
        raise                    

if __name__ == '__main__':
    main()