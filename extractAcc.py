#Extracts validation and test accuracy from logs
import pickle,sys
path = sys.argv[1]
f = open(path,"rb")
log = pickle.load(f)

for i in range(0,len(log)):
	print(",".join([str(i) for i in log[i]["modelParams"]["FilterSizes"]])+" & "+str(100*log[i]["valAcc"])+"\%\\\\\\hline")
print("------")
for i in range(0,len(log)):
        print(str(log[i]["modelParams"]["NumFilters"])+" & +  "+str(100*log[i]["valAcc"]) + "\% & " + str(log[i]['Time']) +"\\\\\\hline")

