 #!/bin/bash

baselogPath=pbsLog/scripts/prova_galileo
mkdir -p $baselogPath
for entry in scripts/prova_galileo/*.pbs
do
        touch pbsLog/$entry
        echo $entry
        echo pbsLog/$entry
        qsub -o logprova/$entry $entry
done
