 #!/bin/bash
baselogPath=logprova/scripts/prova
mkdir -p $baselogPath
for entry in scripts/prova/*.sh
do	
	touch logprova/$entry
	#qsub $entry > logprova/&entry
	echo $entry
	echo logprova/$entry
	$entry > logprova/$entry 2>&1
done

