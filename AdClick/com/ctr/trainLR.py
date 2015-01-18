'''
Created on 01-Jan-2015

@author: niloygupta
'''


import psycopg2
import sys
import numpy as np
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import pickle
from sklearn import linear_model

db_conn = None;

def getRFModel(train,target):
        #rf = RandomForestClassifier(n_estimators=50,max_depth=25)
        rf = AdaBoostClassifier(n_estimators=100)
        rf.fit(train, target)
        return rf
    
def getRFModelOutput(train,models):
        #rf = RandomForestClassifier(n_estimators=50,max_depth=25)
        X = np.array([])
        i = 0
        for model in models:
            p = model.predict(train)
            if i == 0:
                X = np.hstack((X, np.array(p)))
                i = i + 1
            else:
                X = np.vstack((X, np.array(p)))
            
        return X
    
def trandformLabels(train,labels):
        for i in range(0,7):
            train[:,i+2] = labels[i].transform(train[:,i+2])
        return train

def main():
    global db_conn

    try:
        db_conn = psycopg2.connect(database='adclick', user='postgres', port='5433')
        cur = db_conn.cursor()
        labels = pickle.load( open( "label.p", "rb" ) )
        models = pickle.load( open( "models.p", "rb" ) )
        clf = linear_model.SGDClassifier()
        clicked = np.array(['0','1'])
        for i in range(0,500):
            cur.execute("select click,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21 from adclickdata where batch_num=%s" %(i))
            queryResult = cur.fetchall()
            dataChunk  = np.array(queryResult)
            target = dataChunk[:,0]
            train = scipy.delete(dataChunk,0,1)
            train = trandformLabels(train,labels)
            X = getRFModelOutput(train,models)
            X = np.transpose(X)
            clf.partial_fit(X, target,classes=clicked)
            
        pickle.dump( clf, open( "lRegModel.p", "wb" ) )
        
        db_conn.close()
        
    except:
        print "Unexpected error:", sys.exc_info()[0]    
        if (db_conn):
            db_conn.close()
            
        raise                    

if __name__ == '__main__':
    main()