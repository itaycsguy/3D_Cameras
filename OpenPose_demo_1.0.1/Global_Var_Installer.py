import os

class Global_Var_Installer():
	@staticmethod
	def install():
		print()
		dir = str(os.getcwd())
		if os.path.exists(dir):
			print("> Found: " + dir)
			os.system("SETX CUSTOM_OPEN_POSE " + dir)
		images_buffer = dir + "\\images_buffer"
		if os.path.exists(images_buffer):
			print("> Found: " + images_buffer)
			os.system("SETX IMAGES_BUFFER " + images_buffer)
		output_values = dir + "\\output_values"
		if os.path.exists(output_values):
			print("> Found: " + output_values)
			os.system("SETX OUTPUT_VALUES " + output_values)
		print("\n> Environment variables are installed well.")
		
		
	
if __name__ == "__main__":
	Global_Var_Installer.install()