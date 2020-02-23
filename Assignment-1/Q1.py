# I already download the data as 'iris.data'
data_file = open('iris.data', 'r')
data = data_file.read().splitlines()
data_vector = []
for line in data:
	data_vector.append(line.split(","))
out_file = open('iris-svm-input.txt','w')
data_vector.pop(-1)
for item in data_vector:
		if (item[4] == "Iris-setosa"):
			label = 1
		elif (item[4] == "Iris-versicolor"):
			label = 2
		else:
			label = 3
		if (float(item[0]) == 0):
			out_file.write(str(label)+" 2:"+item[1]+" 3:"+item[2]+" 4:"+item[3]+"\n")
		elif (float(item[1]) == 0):
			out_file.write(str(label)+" 1:"+item[0]+" 3:"+item[2]+" 4:"+item[3]+"\n")
		elif (float(item[2]) == 0):
			out_file.write(str(label)+" 1:"+item[0]+" 2:"+item[1]+" 4:"+item[3]+"\n")
		elif (float(item[3]) == 0):
			out_file.write(str(label)+" 1:"+item[0]+" 2:"+item[1]+" 3:"+item[2]+"\n")
		else:
			out_file.write(str(label)+" 1:"+item[0]+" 2:"+item[1]+" 3:"+item[2]+" 4:"+item[3]+"\n")
