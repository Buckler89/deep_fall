 #!/bin/bash
baselogPath=logprova/scripts/prova_galileo
mkdir -p $baselogPath
for entry in scripts/prova_galileo/*.pbs
do
        touch logprova/$entry
        #qsub $entry > logprova/&entry
        echo $entry
        echo logprova/$entry
        qsub -o logprova/$entry $entry
done
