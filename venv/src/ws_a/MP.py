# -*- coding: utf-8 -*-
import re
import math

# 定义词典word_dic
word_dic = {}
word_count_dic = {}
word_probability = {}
count_total = 0
# 定义更新次数
count = 0


# 获取字典：词:词频-总词数-词频
def get_word_dic():
	global word_dic
	global word_count_dic
	global word_probability
	global count_total
	# 定义词频字典，概率字典，总词数
	with open("../词性标注%40人民日报199801.txt", 'r', encoding='utf-8') as file:
		train_txt = file.read()
	for line in train_txt.split("\n"):
		for part in line.split("  "):
			part = part[0: part.find("/")]
			count_total += 1
			if part in word_count_dic:
				word_count_dic[part] += 1
			else:
				word_count_dic[part] = 1
	# 去除词性标注
	with open("../train%40人民日报199801.txt", 'w') as file:
		for line in train_txt.split("\n"):
			for part in line.split("  "):
				line = line.replace(part, part[0: part.find("/")])
			file.write(line)
			file.write("\n")
	# 将词典写入文件
	with open("../word_dic.txt", 'w') as file:
		for key, value in word_count_dic.items():
			p ='{:.8f}'.format(value/count_total)
			word_probability[key] = p
			file.write('{key}:{value1}  {value2}  {value3}\n'.format(key=key, value1=value, value2=count_total, value3=p))
			keyAll = str(value) + '  ' + str(count_total) + '  ' + str(p)
			word_dic[key] = keyAll
	return word_dic


# 当某个字不在词典中时，加入。此时更新词典
def update_word_dic(temp):
	global count
	global count_total
	global word_dic
	global word_count_dic
	global word_probability
	global count_total
	path = "../word_dic"+str(count)+".txt"
	count += 1
	count_total += 1
	# 将新词加入，并写入词典文件
	word_count_dic[temp] = 1
	with open(path, 'w') as file:
		for key, value in word_count_dic.items():
			if key != temp:
				word_count_dic[key] += 1
				value = word_count_dic[key]
			# 概率保留8位小数点
			p ='{:.8f}'.format(value/count_total)
			word_probability[key] = p
			file.write('{key}:{value1}  {value2}  {value3}\n'.format(key=key, value1=value, value2=count_total, value3=p))
			keyAll = str(value) + '  ' + str(count_total) + '  ' + str(p)
			word_dic[key] = keyAll
	return word_dic


# 获取所有候选词，并存入字典 {"出现位置 候选词"：[左邻词,...]}，左邻词初始为[]
# 通过FMM+BMM处理交集歧义
def get_candidate(sentence, max_word_list_count):
	candidate_word_dic = {}
	len = sentence.__len__()
	# 根据最大划分长度，来切分词
	# 前向最大匹配FMM
	j = 0
	while j < len:
		for i in range(max_word_list_count, 0, -1):
			if i + j > len:
				if sentence[j: len] != '':
					if sentence[j: len] in word_dic.keys():
						candidate_word_dic[str(j) + " " + sentence[j: len]] = []
						j = len
						break
					else:
						if i == 1:
							# 更新词典
							update_word_dic(sentence[j: len])
							candidate_word_dic[str(j) + " " + sentence[j: len]] = []
							j = len
							break
			else:
				if sentence[j: j + i] in word_dic.keys():
					candidate_word_dic[str(j) + " " + sentence[j: j + i]] = []
					j = j + i
					break
				else:
					# 如果key中没有这个字，则加入
					if i == 1:
						# 更新词典
						update_word_dic(sentence[j: j + i])
						candidate_word_dic[str(j) + " " + sentence[j: j + i]] = []
						j = j + 1
						break
	# 后向最大匹配BMM
	j = len
	while j > 0:
		for i in range(max_word_list_count, 0, -1):
			if j - i < 0:
				if sentence[j: len] != '':
					if sentence[j: len] in word_dic.keys():
						candidate_word_dic[str(j) + " " + sentence[j: len]] = []
						j = -1
						break
					else:
						if i == 1:
							# 更新词典
							update_word_dic(sentence[j: len])
							candidate_word_dic[str(j) + " " + sentence[j: len]] = []
							j = -1
							break
			else:
				if sentence[j - i: j] in word_dic.keys():
					candidate_word_dic[str(j-i) + " " + sentence[j - i: j]] = []
					j = j - i
					break
				else:
					if i == 1:
						# 更新词典
						update_word_dic(sentence[j - i: j])
						candidate_word_dic[str(j) + " " + sentence[j - i: j]] = []
						j = j - i
						break
	return candidate_word_dic


# 找到所有候选词的左邻词LAW（left adjacent word），存入字典 {"出现位置 词":[左邻词,左邻词...]}
def get_law_dic(sentence, candidate_word_dic):
	law_dic = {}
	for key, value in candidate_word_dic.items():
		law_dic[key] = []
		# 从该词前面找左邻词。先找到该词出现的位置。如果某词中包含该词的前一个字，并且某词的index正确，则为左邻词
		index = int(key.split(" ")[0])
		if index == 0:
			law_dic[key] = []
		else:
			temp = sentence[(index-1): index]
			for now in candidate_word_dic.keys():
				current_index = int(now.split(" ")[0])
				current_word = now.split(" ")[1]
				len = current_word.__len__()
				if current_word[len-1:len] == temp and current_index + len == index:
					if key in law_dic:
						law_dic[key].append(now)
					else:
						law_dic[key].append(now)
	return law_dic


# 计算二元组的对数概率。并使用加1平滑
def calculate_bigram_probability(word1, word2, len):
	with open("../train%40人民日报199801.txt", 'r') as file:
		train_text = file.read()
		# 对于例如（今天 天气）这样的二元组计算累积概率
		count_1 = train_text.count(word1 + "  " + word2)
		count_2 = train_text.count(word1 + "  ")
		# 即条件概率
		probability_a_b = (count_1 + 1) / (count_2 + len)
		log_probability = math.log(probability_a_b)
	return log_probability


# 计算累积概率，并找出最佳左邻词。此处加入平滑。返回最佳左邻词的词典{"出现位置 词"：{"最佳左邻词"：累计概率 }}
def get_best_law(law_dic):
	best_law_dic = {}
	a_probability_log = {}
	# 对左邻词列表的key中的出现位置进行排序，然后计算累计概率
	temp_list = {}
	for current_word in law_dic.keys():
		if int(current_word.split(" ")[0]) in temp_list.keys():
			temp_list[int(current_word.split(" ")[0])].append(current_word)
		else:
			temp_list[int(current_word.split(" ")[0])] = [current_word]
	a = list(temp_list.keys())
	a.sort()
	for key in a:
		for word in temp_list[key]:
			current_word = word
			len = law_dic.__len__()
			# 句首
			if law_dic[current_word] == []:
				a_probability_log[current_word] = math.log(float(word_probability[current_word.split(" ")[1]]))
				best_law_dic[current_word] = []
			else:
				max = -9999999999
				for temp_word in law_dic[current_word]:
					t = float(a_probability_log[temp_word]) + calculate_bigram_probability(temp_word.split(" ")[1], current_word.split(" ")[1], len)
					if max < t:
						max = t
						a_probability_log[current_word] = max
						best_law_dic[current_word] = temp_word
	return best_law_dic, a_probability_log


# 反向查找最佳左邻词
def get_seg_result(sentence, best_law_dic, a_probability_log):
	# 先找到最后一个词
	word = sentence[-1]
	result = ''
	if word != '' and word != "\n":
		last_word = ''
		min_probability = 0.0
		for key, value in best_law_dic.items():
			current_word = key.split(" ")[1]
			probability = a_probability_log[key]
			if current_word[-1] == word and min_probability > probability:
				min_probability = probability
				last_word = key
		result = last_word.split(" ")[1]
		best_left_word = best_law_dic[last_word]
		while best_left_word != []:
			result = best_left_word.split(" ")[1] + " " + result
			best_left_word = best_law_dic[best_left_word]
	return result


# 定义分词算法
def segmentation(sentence):
	# 定义划分最大词长度
	max_word_list_count = 10
	candidate_word_dic = get_candidate(sentence, max_word_list_count)
	law_dic = get_law_dic(sentence, candidate_word_dic)
	best_law_dic, a_probability_log = get_best_law(law_dic)
	result = get_seg_result(sentence, best_law_dic, a_probability_log)
	return result


get_word_dic()

# 使用正则表达式。将所有句子根据各种标点符号划分。有可能句子前后都有标点。则判断整行前有没有标点，剩下的在句子后判断
pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|！| |…|（|）|●'

with open("../test.txt", "r", encoding="gbk") as file:
	with open("../2018110663.txt", 'w') as result_file:
		test_text = file.read()
		# 按照每一行处理
		for line in test_text.split("\n"):
			# 如果该行不为空，且第一位为标点，则写入文件
			if line != "" and pattern.find(line[0]) != -1:
				result_file.write(line[0] + " ")
			sentence_list = re.split(pattern, line[0: line.__len__()])
			for sentence in sentence_list:
				if sentence == "" or sentence == "\n":
					continue
				result_file.write(segmentation(sentence))
				index = line.find(sentence)
				len = sentence.__len__()
				punctuation = line[index+len: index+len+1]
				if punctuation != " ":
					result_file.write(" ")
					result_file.write(punctuation)
				result_file.write(" ")
			result_file.write("\n")
