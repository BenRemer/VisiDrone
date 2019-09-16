#!usr/bin/env python3
import zeep
import requests
from requests.auth import HTTPDigestAuth
from onvif import ONVIFCamera, ONVIFService
import shutil

# A Monkey Patch provided by an online forum
def zeep_pythonvalue(self, xmlvalue):
    return xmlvalue
zeep.xsd.simple.AnySimpleType.pythonvalue = zeep_pythonvalue

# Returns a string of the current data and time to be used as a filename
def getFileName():
    dt = mycam.devicemgmt.GetSystemDateAndTime()
    tz = dt.TimeZone
    year = dt.UTCDateTime.Date.Year
    month = dt.UTCDateTime.Date.Month
    date = dt.UTCDateTime.Date.Day
    hour = dt.UTCDateTime.Time.Hour
    minute = dt.UTCDateTime.Time.Minute
    second = dt.UTCDateTime.Time.Second
    filename = 'Images/%s-%s-%s-%s-%s-%s-%s.jpg' % (year, month, date, hour, minute, second, str(i))
    return filename

# Note credentials for login. 
mycam = ONVIFCamera('169.254.126.176', 80, 'administrator', 'password')
media = mycam.create_media_service()
profiles = media.GetProfiles()
    
i = 0 # Corresponds to camera head
for p in profiles: # Each profile has some output. There is a single 'Primary' profile for each camera head.
    print(p.token)
    if ('Primary' in p.token):
        snapshot = media.GetSnapshotUri({'ProfileToken' : p.token})
        print(snapshot['Uri']) # URL from which we download a still

        filename = getFileName()

        r = requests.get(snapshot['Uri'], auth=HTTPDigestAuth('administrator', 'password'), stream=True)
        with open(filename, 'wb') as out_file: # Copy the http response into a jpg file with file name in a dateTime format.
            shutil.copyfileobj(r.raw, out_file)
        del r
        i += 1 # Iterate for each camera head's image.
