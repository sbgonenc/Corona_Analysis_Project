'''
def prot_parser(fasta_file):
	#fasta dosyasını parse'lar, pandas dataframe (ülke-tarih-protein adı- gisaidkodu-sequence) return eder
	dict_for_seq = {}
	with open(fasta_file) as f:
		for lines in f:
			line = lines.strip()
			if line.startswith('>'):
				headers_list = line.split('|')
				protein_name = headers_list[0][1:]
				date = headers_list[2]
				EPI_code = headers_list[3]
				country = headers_list[-1]
			else:
				sequences = line
				if country not in dict_for_seq:
					dict_for_seq[country] = [{date: {protein_name: len(sequences)}}]
				else: dict_for_seq[country].append({date: {protein_name: len(sequences)}})

		return dict_for_seq

pprint(prot_parser(prot_file))
'''


'''
def str_float_convert(str_type, index):
	import re

	try:
		f_type = float(str_type)
		if f_type: return f_type

	except:
			s_list = str_type.split('.')
			
			
			if len(s_list) > 3:
				for _ in range(len(s_list)):
					if '000' == s_list[-1]: s_list.pop()
					if '990' == s_list[-1] or '999' == s_list[-1]: s_list.pop()

			s_str = ''.join(s_list)
			if s_str[-1] == '0': s_str = s_str[:-1]
			s_str = s_str[0:3] + '.' + s_str[3:]
			s_str = round(float(s_str), 3)
			return s_str'''