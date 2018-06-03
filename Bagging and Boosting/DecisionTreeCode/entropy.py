from math
import numpy
import log
import operator

def Entropy(tattr, data, attribute):
    
    fre_value = {}
    entropy_data = 0.0

    #find index of the target attribute

    i=0
    for value in attribute:
        if(tattr == value)
        ++i
   
    # Farquency of each target attribute

    if (fre_value.has_key(record[tattr])):
    fre_value[record[tattr]] += 1.0
	else:
    fre_value[record[tattr]] = 1.0

    # Entropy calculation for the data of the target attribute
   	
   	for freq in fre_value.values():
    entropy_data += (-freq/len(data))* math.log(freq/len(data),2)
    return entropy_data

def informationgain(tattr, data, attribute)

	fre_value = {}
	entropy_subset = 0.0
	difference = 0.0

	# frequency of values of the target attribute
    
    for value in data:
    if (fre_value.has_key(record[attribute])):
        fre_value[record[attribute]] += 1.0
    else:
        fre_value[record[attribute]] = 1.0

    #sum of entropy of each of the partitioned subsets.
    
    for items in fre_value.keys():
    val_prob = fre_value[items]/ sum(val_freq.items())
    data_subset = [record for record in data if record[attribute] == items]
    p_data = [entry for entry in data if entry[i] == val]
    entropy_subset += val_prob * Entropy(tattr, data, p_data)
   
    #return the difference between the entropy of the choosen attribute and the entropy of the whole set
    
    difference = Entropy(tattr, data) - entropy_subset
	return difference