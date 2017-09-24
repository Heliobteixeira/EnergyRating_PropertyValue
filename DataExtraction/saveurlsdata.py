import mechanize
import cookielib
from copy import copy
from time import sleep
import json
import re
import os.path

def loadjson(jsonfile):
    emptyjson={}
    try:
        with open(jsonfile, 'r') as file:
            return json.load(file)
    except:
        print 'Ficheiro '+jsonfile+' vazio'
        return emptyjson

def savedata(id, html):
    fname=str(id)+'.html'
    
    with open(fname, 'w') as file:    
        file.write(html)

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
    br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=10000)
    br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]    
    return br,cj


    

print 'Started...'
urlsfile='urlsdb.json'

urls=loadjson(urlsfile)
for id, data in urls.items():
    url=data['url']
    if idexists(id):
        print id+' already saved. Continuing to the next one...'
        continue
    data={}
    print 'Parsing:'+url
    (br, cj)=newconfbrowser()
    try:
        html=br.open(url).read()
    except:
        print id+' unavailable'
        continue
    else:

    soup=BeautifulSoup(html)
    

        
        
