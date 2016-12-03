#!/bin/bash

echo 'answers-forums'
forum=$(perl correlation-noconfidence.pl STS.gs.answers-forums.txt sys.forum)
res1=$(echo $forum | grep -o '0\.[0-9]\+')
echo $res1

echo 'answers-students'
student=$(perl correlation-noconfidence.pl STS.gs.answers-students.txt sys.students)
res2=$(echo $student | grep -o '0\.[0-9]\+')
echo $res2

echo 'belief'
student=$(perl correlation-noconfidence.pl STS.gs.belief.txt sys.belief)
res3=$(echo $student | grep -o '0\.[0-9]\+')
echo $res3

echo 'headlines'
student=$(perl correlation-noconfidence.pl STS.gs.headlines.txt sys.headlines)
res4=$(echo $student | grep -o '0\.[0-9]\+')
echo $res4

echo 'images'
student=$(perl correlation-noconfidence.pl STS.gs.images.txt sys.images)
res5=$(echo $student | grep -o '0\.[0-9]\+')
echo $res5

echo 'Mean'
# echo $((0+${res1#0}+${res2#0}))
# echo "($res1 + $res2 + $res3 + $res4 + $res5)" | bc
calc=$(echo "$res1 + $res2 + $res3 + $res4 + $res5"|bc)
bc <<< "scale=5;$calc/5.0"
