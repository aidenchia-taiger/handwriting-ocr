import enchant
import re
import argparse

def analyze(file, name):

	reportFound = False
	numericCER = 0
	numericTotal = 0
	numericList = []
	wordCER = 0
	wordTotal = 0
	wordList = []
	nricCER = 0
	nricTotal = 0
	nricList = []
	dateCER = 0
	dateTotal = 0
	dateList = []
	asianCER = 0
	asianTotal = 0
	asianList = []
	addressCER = 0
	addressTotal = 0
	addressList = []
	othersCER = 0
	othersTotal = 0
	othersList = []

	for line in file.readlines():
		if not reportFound:
			if '[MODEL]' in line:
				line = line.strip().split('| ')
				if len(line) < 2:
					continue
				else:
					if name == line[1]:
						reportFound = True
		else:
			if '[INFO]' in line:
				break
			split = line.strip().split()
			# get the number X from [ERRX] which is the CER
			cer = int(split[0][4:-1]) if 'ERR' in split[0] else 0 
			gt = split[1]
			if classOf(gt) == 'numeric':
				numericCER += cer
				numericTotal += len(gt)
				numericList.append(line.strip())
			elif classOf(gt) == 'word':
				wordCER += cer
				wordTotal += len(gt)
				wordList.append(line.strip())
			elif classOf(gt) == 'nric':
				nricCER += cer
				nricTotal += len(gt)
				nricList.append(line.strip())
			elif classOf(gt) == 'date':
				dateCER += cer
				dateTotal += len(gt)
				dateList.append(line.strip())
			elif classOf(gt) == 'asian':
				asianCER += cer
				asianTotal += len(gt)
				asianList.append(line.strip())
			elif classOf(gt) == 'address':
				addressCER += cer
				addressTotal += len(gt)
				addressList.append(line.strip())
			elif classOf(gt) == 'others':
				othersCER += cer
				othersTotal += len(gt)
				othersList.append(line.strip())


	print('\n ===================== NUMERIC LIST =========================================')
	[print(x) for x in numericList]
	
	print('\n ===================== WORD LIST ====================================')
	[print(x) for x in wordList]

	print('\n ===================== ADDRESS LIST ====================================')
	[print(x) for x in addressList]
	
	print('\n ===================== ASIAN LIST ====================================')
	[print(x) for x in asianList]

	print('\n ===================== OTHERS LIST ====================================')
	[print(x) for x in othersList]

	print('\n ===================== CER ====================================')

	print('\n[INFO] Numeric CER = {:.{}f}% \t| Num = {}'.format(numericCER * 100 / numericTotal,2, len(numericList)))
	print('[INFO] Word CER = {:.{}f}% \t| Num = {}'.format(wordCER * 100 / wordTotal,2, len(wordList)))
	print('[INFO] NRIC CER = {:.{}f}% \t| Num = {}'.format(nricCER * 100 / nricTotal,2, len(nricList)))
	print('[INFO] Address CER = {:.{}f}% \t| Num = {}'.format(addressCER * 100 / addressTotal,2, len(addressList)))
	print('[INFO] Asian CER = {:.{}f}% \t| Num = {}'.format(asianCER * 100 / asianTotal,2, len(asianList)))
	print('[INFO] Others CER = {:.{}f}% \t| Num = {}'.format(othersCER * 100 / othersTotal,2, len(othersList)))


def classOf(gt):
	'classify ground truth into numeric, word, NRIC, date, or asian'
	d = enchant.Dict("en_GB")
	if is_numeric(gt):
		#print(gt, 'numeric')
		return 'numeric'

	elif d.check(gt):
		#print(gt, 'word')
		return 'word'

	elif any(char.isdigit() for char in gt) and gt[0] == 'S' or gt[0] == 's':
		#print(gt, 'nric')
		return 'nric'

	elif '/' in gt and [ x.isdigit() for x in gt.split('/')]:
		#print(gt, 'date')
		return 'date'

	elif is_address(gt):
		#print(gt, 'address')
		return 'address'

	elif gt.isalpha():
		#print(gt, 'asian')
		return 'asian'
	else:
		#print(gt, 'others')
		return 'others'

def is_address(gt):
	regex = re.compile('[@_#()]')
	if regex.search(gt) == None:
		return False
	else:
		return True

def is_numeric(gt):
	if ',' in gt:
		if [x.isdigit() for x in gt.split(',')]:
			return True
	elif gt.isdigit():
		return True

	else:
		return False


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--i', help="name of model found in report.txt e.g. cherry trained on 54 epochs on new_data")
	args = parser.parse_args()

	file = open('../model/report.txt')
	analyze(file, args.i)
	file.close()

if __name__=='__main__':
	main()