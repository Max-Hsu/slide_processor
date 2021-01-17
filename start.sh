#! /bin/bash
if [ -d "$1" ]
then
    echo "directory mode"
    todo=$(ls $1| sed "s@\(.*\)@$1\/\1@" |grep -e".*png$" -e ".*jpg$" -e ".*tif$")
    shift
    python3 do.py $todo $@
elif [ -f "$1" ]
then
    echo "file mode"
    python3 do.py $@
elif [ "$1" == "file" ]
then
    echo "batch file mode"
    shift
    todo=$(cat $1   |grep -e".*png$" -e ".*jpg$" -e ".*tif$")
    shift
    python3 do.py $todo $@
else
    echo "current directory mode"
    todo=$(ls | grep -e".*png$" -e ".*jpg$" -e ".*tif$")
    python3 do.py $todo $@
fi
