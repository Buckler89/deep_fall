 #!/bin/bash
case="case1"
baselogPath=pbsLog/scripts/case
mkdir -p $baselogPath
for entry in scripts/case/*.pbs
do
        touch pbsLog/$entry
        echo $entry
        echo pbsLog/$entry
        qsub -o pbsLog/$entry $entry
done
