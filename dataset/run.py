import subprocess
import re

pipe = subprocess.Popen(["./correlation-noconfidence.pl","./STS.gs.answers-forums.txt","./sys.forum"],stdout=subprocess.PIPE)
result = pipe.stdout.read()
result1 = re.findall("\d+\.\d+", result)[0]

pipe = subprocess.Popen(["./correlation-noconfidence.pl","./STS.gs.answers-students.txt","./sys.students"],stdout=subprocess.PIPE)
result = pipe.stdout.read()
result2 = re.findall("\d+\.\d+", result)[0]

pipe = subprocess.Popen(["./correlation-noconfidence.pl","./STS.gs.belief.txt","./sys.belief"],stdout=subprocess.PIPE)
result = pipe.stdout.read()
result3 = re.findall("\d+\.\d+", result)[0]


pipe = subprocess.Popen(["./correlation-noconfidence.pl","./STS.gs.headlines.txt","./sys.headlines"],stdout=subprocess.PIPE)
result = pipe.stdout.read()
result4 = re.findall("\d+\.\d+", result)[0]


pipe = subprocess.Popen(["./correlation-noconfidence.pl","./STS.gs.images.txt","./sys.images"],stdout=subprocess.PIPE)
result = pipe.stdout.read()
result5 = re.findall("\d+\.\d+", result)[0]


s = [result1, result2, result3, result4, result5]
s = [round(float(i), 4) for i in s]
print "answers-forums: " + str(s[0])
print "answers-students: " + str(s[1])
print "belief: " + str(s[2])
print "headlines: " + str(s[3])
print "images: " + str(s[4])


print "mean: " + str(round(sum(s) / float(len(s)), 4))