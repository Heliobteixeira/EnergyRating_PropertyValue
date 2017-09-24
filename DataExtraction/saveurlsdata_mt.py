#!/usr/bin/env python
# -*- coding: latin-1 -*-

import mechanize
import cookielib
from time import sleep
import json
import pdb
import re
from time import time
from Queue import Queue, Empty
from threading import Thread
from threading import Lock
import codecs
import os.path


STARTTIME=time()
COUNT=0
LASTSAVE=time()
MAXSAVEINTERVAL=60  # seconds
TIPOSPROPRIEDADE=['Apartamento', 'Moradia/Vivenda', 'Duplex']


outputdir='output/html/'
jsonfile='mined_urls_total.json'


def loadjson(jsonfile):
    emptyjson={}
    try:
        loadedfile=file(jsonfile, 'r')
        filetostring=loadedfile.read().decode("utf-8-sig")
        result=json.loads(filetostring)
        return result
    except IOError as e:
        print 'Ficheito %s vazio' % jsonfile
        return emptyjson
    except Exception as e:
        print e
        #pdb.set_trace()

def savedata(id, html):
    fname='output/html/'+str(id)+'.html'
    
    with open('workfile', 'w') as output_file:
        output_file.write(html)



def idexists(id):
    fname=str(id)+'.html'
    return os.path.isfile(fname)

def newconfbrowser():
    br = mechanize.Browser()
    cj = cookielib.LWPCookieJar()
    br.set_cookiejar(cj)
    br.set_handle_equiv(True)
    br.set_handle_redirect(True)
    br.set_handle_robots(False)
    br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=10)
    br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]    
    return br,cj

class ThreadMechanizeUrl(Thread):
    
    def __init__ (self, queue, lock):
        self.queue=queue
        self.lock=lock
        Thread.__init__ (self)

    def run(self):
        global STARTTIME
        global COUNT
        global JSONDATA
        global LASTSAVE
        global URLPREFIX

        while True:
            (id, url)=self.queue.get()

            elapsed=time()-STARTTIME
            parspersec=COUNT/elapsed
            # data={}
            print "%s urls in %0.2f min " % (COUNT, elapsed/60.0) + " | %0.2furls/sec | " % parspersec +'Parsing:'+id
            (br, cj)=newconfbrowser()
            
            try:
                html=br.open(url).read()
                #pdb.set_trace()
            except:
                print url+'!! UNAVAILABLE !!'
                self.queue.task_done()
                continue
            else:       

                if (time()-LASTSAVE) > MAXSAVEINTERVAL:
                    #self.lock.acquire()
                    savedata(id, html)
                    #self.lock.release()
                    LASTSAVE=time()
                    print "Data saved..."

                COUNT+=1
                self.queue.task_done()
        print "End of task"
                  
        
print 'Started...'

queue = Queue()
lock = Lock()

datasourcevenda=loadjson('output/json/mined_venda_filtered.json')
datasourcearr=loadjson('output/json/mined_arrendamento_filtered.json')
#spawn a pool of threads, and pass them queue instance 
for i in range(10):
    t = ThreadMechanizeUrl(queue, lock)
    t.setDaemon(True)
    t.start()
    
for id, data in datasourcevenda.items():
    if idexists(id):
        print id+' already parsed. Continuing to the next one...'
    else:
        try:
            queue.put((id, data['url']))
            #pdb.set_trace()
        except KeyboardInterrupt:
            print "Stopping..."

for id, data in datasourcearr.items():
    if idexists(id):
        print id+' already parsed. Continuing to the next one...'
    else:
        try:
            queue.put((id, data['url']))
            #pdb.set_trace()
        except KeyboardInterrupt:
            print "Stopping..."

queue.join()
print "All DOOOONE"
