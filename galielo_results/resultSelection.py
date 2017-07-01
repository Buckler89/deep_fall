import csv
import numpy as np
import os
import sys
valueLab = dict()
results = []

finalBest = []

case = 'case1_upto_999'#sys.argv[1]
ingored=0;
sourceFile = 'totalReport.csv'
targetFileName = 'bestResults.csv'
with open(os.path.join(case, sourceFile), newline='') as csvfile:
    #read first line: the label of column
    firstLine = csvfile.readline()
    rowval = firstLine.split(',')
    valueLab['id'] = 0
    for i, val in enumerate(rowval[1:]):
        valueLab[val.strip('\n')] = i+1

    #all the line
    lines = csvfile.readlines()
    nline=len(lines)
    for row in lines: #the fist line is already parsed
        rowval = row.split(',')
        rowval[0] = rowval[0].replace('process_', '')
        if int(rowval[0]) < 1000:
            for i, val in enumerate(rowval):
                if i != 0:
                    if float(val) > 1:
                        val = float(val)*0.001
                        print(val)
                try:
                    results.append(float(val))
                except:
                    print(row)

                #result.append({})
            #print(', '.join(row))
        else:
            ingored+=1;
results = np.array(results)
results = results.reshape(nline-ingored, len(valueLab))
print()

fold = 1
for lab in ['AucDevsFold1', 'AucDevsFold2', 'AucDevsFold3', 'AucDevsFold4']:
    bestResults = []
    bestbestResults = []
    bestbestbestResults = []
    bestbestbestbestResults = []
    potentialBestResults = []

    col = results[:, [valueLab[lab]]]
    best = col.max()
    for row in results:
        if row[valueLab[lab]] == best:
            if not bestResults: #the fist time the list is empty
                bestResults.append(row)
            elif not row[valueLab['id']] in [r[valueLab['id']] for r in bestResults]:
                bestResults.append(row)
    #now we have all best result based on AucDevsFoldX

    sublab='f1DevsFold'+str(fold)
    bestResultsNP = np.array(bestResults)
    col = bestResultsNP[:, [valueLab[sublab]]]
    best = col.max()
    for row in bestResultsNP:
        if row[valueLab[sublab]] == best:
            if not bestbestResults: #the fist time the list is empty
                bestbestResults.append(row)
            elif not row[valueLab['id']] in [r[valueLab['id']] for r in bestbestResults]:
                bestbestResults.append(row)
    #now we have all best result based on f1DevsFoldX

    subsublab = 'AucTestFold'+str(fold)
    bestbestResultsNP = np.array(bestbestResults)
    col = bestbestResultsNP[:, [valueLab[subsublab]]]
    best = col.max()
    for row in bestbestResultsNP:
        if row[valueLab[subsublab]] == best:
            if not bestbestbestResults: #the fist time the list is empty
                bestbestbestResults.append(row)
            elif not row[valueLab['id']] in [r[valueLab['id']] for r in bestbestbestResults]:
                bestbestbestResults.append(row)

    subsubsublab = 'f1TestFold'+str(fold)
    bestbestbestResultsNP = np.array(bestbestbestResults)
    col = bestbestbestResultsNP[:, [valueLab[subsubsublab]]]
    best = col.max()
    for row in bestbestbestResultsNP:
        if row[valueLab[subsubsublab]] == best:
            if not bestbestbestbestResults: #the fist time the list is empty
                bestbestbestbestResults.append(row)
            elif not row[valueLab['id']] in [r[valueLab['id']] for r in bestbestbestbestResults]:
                bestbestbestbestResults.append(row)

    #finalBest.append(bestbestbestbestResults)
    finalBest = finalBest + bestbestbestbestResults
    fold += 1

finalBestNP = np.array(finalBest)
finalBest_ = []
for i in finalBest:
  if not i[valueLab['id']] in [f[valueLab['id']] for f in finalBest_]:
    finalBest_.append(i)


for fold in range(1, 5):
    lab = 'f1TestFold'+str(fold)
    col = results[:, [valueLab[lab]]]
    best = col.max()
    for row in results:
        if row[valueLab[lab]] == best:
            # the fist time the potentialBestResults list is empty
            if not potentialBestResults and not row[valueLab['id']] in [r[valueLab['id']] for r in finalBest_]:
                potentialBestResults.append(row)
            elif not row[valueLab['id']] in [r[valueLab['id']] for r in potentialBestResults] and not row[valueLab['id']] in [r[valueLab['id']] for r in finalBest_]:
                potentialBestResults.append(row)


meanBestResults = []
lab = 'f1Final'
col = results[:, [valueLab[lab]]]
best = col.max()
for row in results:
    if row[valueLab[lab]] == best:
        # the fist time the potentialBestResults list is empty
        if not meanBestResults and not row[valueLab['id']] in [r[valueLab['id']] for r in finalBest_]:
            meanBestResults.append(row)
        elif not row[valueLab['id']] in [r[valueLab['id']] for r in meanBestResults] and not row[valueLab['id']] in [r[valueLab['id']] for r in finalBest_]:
            meanBestResults.append(row)

with open(os.path.join(case,targetFileName), mode='w') as file:
    lab='id,AucDevsFold1,AucDevsFold2,AucDevsFold3,AucDevsFold4,AucTestFold1,AucTestFold2,AucTestFold3,AucTestFold4,f1DevsFold1,f1DevsFold2,f1DevsFold3,f1DevsFold4,f1Final,f1TestFold1,f1TestFold2,f1TestFold3,f1TestFold4\n'
    file.write(lab.replace(',','\t'))
    for l in finalBest_:
        for v in l:
            file.write(str(v).replace('.',',')+'\t')
        file.write('\n')

    file.write(lab.replace(',','\t'))
    for l in potentialBestResults:
        for v in l:
            file.write(str(v).replace('.',',')+'\t')
        file.write('\n')

print()
