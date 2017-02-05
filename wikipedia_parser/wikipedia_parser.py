import json, os, wikipedia

class WikipediaParser:

	def __init__(self):
		self.training_pages = self.json_loader("data/wikipedia_training_pages.json")

	def json_loader(self,filename):
		with open(filename) as data_file:
			data = json.load(data_file)
		return data

	def get_polar_training_pages(self):
		return self.training_pages['polar-regions-pages']

	def get_forests_training_pages(self):
		return self.training_pages['forests-pages']

	def get_oceans_training_pages(self):
		return self.training_pages['oceans-pages']

	def get_air_training_pages(self):
		return self.training_pages['air-pages']

	def fill_polar_data(self):
		os.remove('data/polar_data.txt')
		polar_pages = self.get_polar_training_pages()
		polar_file = open('data/polar_data.txt', 'w')
		data = ""
		for page in polar_pages:
			print(self.get_page(page).title)
			data += self.get_page(page).content
		polar_file.write(data)
		polar_file.close()
		return len(polar_pages)

	def fill_forests_data(self):
		os.remove('data/forests_data.txt')
		forests_pages = self.get_forests_training_pages()
		forests_file = open('data/forests_data.txt', 'w')
		data = ""
		for page in forests_pages:
			print(self.get_page(page).title)
			data += self.get_page(page).content
		forests_file.write(data)
		forests_file.close()
		return len(forests_pages)

	def fill_oceans_data(self):
		os.remove('data/oceans_data.txt')
		oceans_pages = self.get_oceans_training_pages()
		oceans_file = open('data/oceans_data.txt', 'w')
		data = ""
		for page in oceans_pages:
			print(self.get_page(page).title)
			data += self.get_page(page).content
		oceans_file.write(data)
		oceans_file.close()
		return len(oceans_pages)

	def fill_air_data(self):
		os.remove('data/air_data.txt')
		air_pages = self.get_air_training_pages()
		air_file = open('data/air_data.txt', 'w')
		data = ""
		for page in air_pages:
			print(self.get_page(page).title)
			data += self.get_page(page).content
		air_file.write(data)
		air_file.close()
		return len(air_pages)

	def fill_all_data_files(self):
		file_count = 0
		file_count += self.fill_polar_data()
		file_count += self.fill_forests_data()
		file_count += self.fill_oceans_data()
		file_count += self.fill_air_data()
		print("All {} data files have been successfully filled.".format(file_count))

	def get_page(self,page):
		return wikipedia.page(page)