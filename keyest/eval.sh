#!/usr/bin/env bash

echo "Giantsteps Validation:"
evaluate key -i ~/data/giantsteps-mtg-key-dataset-augmented/annotations/key/*.0.key $1/val/*.0.key.txt
echo "Billboard Validation:"
evaluate key -i ~/data/mcgill-billboard-augmented/annotations/key/*.0.key $1/val/*.0.key.txt
echo "cmdb Validation:"
evaluate key -i ~/data/classical_music_database-augmented/annotations/key/*.0.key $1/val/*.0.key.txt
echo "Total Validation:"
evaluate key -i ~/data/giantsteps-mtg-key-dataset-augmented/annotations/key/*.0.key \
                ~/data/mcgill-billboard-augmented/annotations/key/*.0.key \
                ~/data/classical_music_database-augmented/annotations/key/*.0.key \
                $1/val/*.0.key.txt

echo "-------------------------"

echo "Giantsteps Test:"
evaluate key -i ~/data/giantsteps-key-dataset/annotations/key/*.key $1/test/*.key.txt
echo "Billboard Test:"
evaluate key -i ~/data/mcgill-billboard-augmented/annotations/key/*.0.key $1/test/*.0.key.txt
echo "cmdb Test:"
evaluate key -i ~/data/classical_music_database-augmented/annotations/key/*.0.key $1/test/*.0.key.txt
echo "Total Text:"
evaluate key -i ~/data/giantsteps-key-dataset/annotations/key/*.key \
                ~/data/mcgill-billboard-augmented/annotations/key/*.0.key \
                ~/data/classical_music_database-augmented/annotations/key/*.0.key \
                $1/test/*.key.txt
