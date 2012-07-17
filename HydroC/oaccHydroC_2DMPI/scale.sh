#!/bin/sh
cp HydroC input ptmp
rm r.txt
for i in 16; do
    (cd ptmp; ccc_mprun -n $i HydroC -i input ) | tee r.txt
done
#EOF
