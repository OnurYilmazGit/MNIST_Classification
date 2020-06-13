import os

def main():
	print('Please press 1 to see KNN analysis on MNIST dataset')
	print('Please press 2 to see Naive-Bayes analysis on MNIST dataset')
	print('Please press 3 to see DecisionTree analysis on MNIST dataset')
	input1 = input('Enter a number here: ')
	if int(input1) == 1:
		os.system('python3 KNN.py')
	elif int(input1) == 2:		
		os.system('python3 naive_bayes.py')
	elif int(input1) == 3:
		os.system('python3 DecisionTreeClassifier.py')
	else: 
		print('You pressed wrong number..')

if __name__ == "__main__":
	main()