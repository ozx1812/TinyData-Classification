#!/bin/bash

# Check if file argument is provided
if [ $# -eq 0 ]; then
  echo "Usage: ./parse_logs.sh <file>"
  exit 1
fi

echo "lr,epochs,batch_size,mean_accuracy,+-" > accuracy_challange2.csv

while read -r line
do
  if [[ "$line" == *"lr="* ]]; then
    lr=$(echo $line | grep -oP '(?<=lr=)[0-9\.]+')
    epochs=$(echo $line | grep -oP '(?<=epochs=)\d+')
    batch_size=$(echo $line | grep -oP '(?<=batch_size=)\d+')
  elif [[ "$line" == *"Acc over 25 instances:"* ]]; then
    mean_accuracy=$(echo $line | grep -oP '(\K\d+\.\d+ \+- \d+\.\d+)' | awk '{print $1}')
    plusminus_accuracy=$(echo $line | grep -oP '(\K\d+\.\d+ \+- \d+\.\d+)' | awk '{print $3}')
    echo "$lr,$epochs,$batch_size,$mean_accuracy,$plusminus_accuracy" >> accuracy_challange2.csv
  fi
done < "${1}"

