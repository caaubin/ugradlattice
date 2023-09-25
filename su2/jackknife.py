import numpy as np 
import statistics as stat

def jack(avgPlaq):
	# avgPlaq: array with the avg plaquettes for each config
	#avgJack: empty array to fill with averages
	#knife: average of avgJack

	avgJack = np.zeros(len(avgPlaq)) 

	for i in range(len(avgPlaq)):

		jack = avgPlaq
		jack = np.delete(jack,i)

		avgJack[i] = np.mean(jack)


	knife = np.mean(avgJack)

	stdKnife = stat.stdev(avgJack,knife) 

	print("jackknife: %f +/- %f" % (knife,stdKnife))

	return knife, stdKnife




