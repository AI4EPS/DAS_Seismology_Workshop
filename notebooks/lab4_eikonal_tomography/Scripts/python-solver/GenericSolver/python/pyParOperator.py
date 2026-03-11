#Module containing the definition of the operator class defined by parameter files
#It takes vector objects from the pyVector class
import pyOperator as pyop
import pyVector as pyvec
import sys_util


#Auxiliary commands
def command_read(cmd_file):
	"""Function to read parameter files correctly"""
	cmd=[]
	with open(cmd_file) as file:
		for line in file:
			if(line[0:4]=="RUN:"):
				cmd_tmp=line.split("RUN:")[1]
				cmd_tmp=cmd_tmp.replace('\r','')
				cmd.append(cmd_tmp.strip('\n'))
	return cmd

def set_random_names(cmd):
	"""Function to change tmp_rand_name[0-9]+ to random temporary name"""
	ind=0
	input_name="tmp_random_name%s"
	while any(input_name%ind in s for s in cmd):
		rand_input_name=sys_util.rand_name(12)
		for ii in range(0,len(cmd)):
			cmd[ii]=cmd[ii].replace(input_name%ind,rand_input_name)
		ind+=1
	return cmd

class parOperator(pyop.Operator):
	"""Class for operators defined by parameter files which runs system commands"""

	def __init__(self,model,data):
		"""Domain and Range of operator; call set_forward and set_adjoint to set actual operators"""
		#Setting Domain and Range size
		self.setDomainRange(model,data)
		return

	def set_forward(self,cmd_file,input_file,output_file,input_m0_file=None,input_aux=[]):
		"""Class to set parameter files for forward operator"""
		self.cmd_file = cmd_file
		self.input_fwd = input_file
		self.input_m0_fwd = input_m0_file
		self.output_fwd = output_file
		self.input_aux_fwd = input_aux #other auxiliary files
		cmd_fwd = command_read(cmd_file)
		#Setting random names if tmp_rand_name[0-9]+ are used
		self.cmd_fwd = set_random_names(cmd_fwd)
		return

	def set_input_output_fwd(self,input=None,output=None,input_m0=None,input_aux=[]):
		"""Method to change name of operator input and/or output"""
		if(input!=None):
			#Changing input file name
			for ii in range(0,len(self.cmd_fwd)):
				self.cmd_fwd[ii]=self.cmd_fwd[ii].replace(self.input_fwd,input)
			self.input_fwd=input
		if(output!=None):
			#Changing output file name
			for ii in range(0,len(self.cmd_fwd)):
				self.cmd_fwd[ii]=self.cmd_fwd[ii].replace(self.output_fwd,output)
			self.output_fwd=output
		if((input_m0!=None) and (self.input_m0_fwd!=None)):
			#Changing input_m0 file name
			for ii in range(0,len(self.cmd_fwd)):
				self.cmd_fwd[ii]=self.cmd_fwd[ii].replace(self.input_m0_fwd,input_m0)
			self.input_m0_fwd=input_m0
		elif((input_m0!=None) and (self.input_m0_fwd==None)):
			raise ValueError("Changing input_m0 without inizitializing it")
		if(not isinstance(input_aux,list)): raise TypeError("Provived input_aux not a list: %s"%(input_aux))
		for idx, ifile in enumerate(input_aux):
			#Changing input_aux[idx] file name if index smaller than internal list length
			if(idx<len(self.input_aux_fwd)):
				for ii in range(0,len(self.cmd_fwd)):
					self.cmd_fwd[ii]=self.cmd_fwd[ii].replace(self.input_aux_fwd[idx],ifile)
				self.input_aux_fwd[idx]=ifile
			else:
				print("WARNING! %s outside of internal list %s"%(ifile,self.input_aux))
		return

	def set_adjoint(self,cmd_file,input_file,output_file,input_m0_file=None,input_aux=[]):
		"""Class to set parameter files for adjoint operator"""
		self.cmd_file = cmd_file
		self.input_adj = input_file
		self.input_m0_adj = input_m0_file
		self.output_adj = output_file
		self.input_aux_adj = input_aux #other auxiliary files
		cmd_adj = command_read(cmd_file)
		#Setting random names if tmp_rand_name[0-9]+ are used
		self.cmd_adj = set_random_names(cmd_adj)
		return

	def set_input_output_adj(self,input=None,output=None,input_m0=None,input_aux=[]):
		"""Method to change name of operator input and/or output"""
		if(input!=None):
			#Changing input file name
			for ii in range(0,len(self.cmd_adj)):
				self.cmd_adj[ii]=self.cmd_adj[ii].replace(self.input_adj,input)
			self.input_adj=input
		if(output!=None):
			#Changing output file name
			for ii in range(0,len(self.cmd_adj)):
				self.cmd_adj[ii]=self.cmd_adj[ii].replace(self.output_adj,output)
			self.output_adj=output
		if((input_m0!=None) and (self.input_m0_adj!=None)):
			#Changing input_m0 file name
			for ii in range(0,len(self.cmd_adj)):
				self.cmd_adj[ii]=self.cmd_adj[ii].replace(self.input_m0_fwd,input_m0)
			self.input_m0_adj=input_m0
		elif((input_m0!=None) and (self.input_m0_adj==None)):
			raise ValueError("Changing input_m0 without inizitializing it")
		if(not isinstance(input_aux,list)): raise TypeError("Provived input_aux not a list: %s"%(input_aux))
		for idx, ifile in enumerate(input_aux):
			#Changing input_aux[idx] file name if index smaller than internal list length
			if(idx<len(self.input_aux_adj)):
				for ii in range(0,len(self.cmd_adj)):
					self.cmd_adj[ii]=self.cmd_adj[ii].replace(self.input_aux_adj[idx],ifile)
				self.input_aux_adj[idx]=ifile
			else:
				print("WARNING! %s outside of internal list %s"%(ifile,self.input_aux))
		return

	def run_operator(self,cmd,print_cmd=False,print_output=False):
		"""Method to run operator"""
		for cmd_to_run in cmd:
			sys_util.RunShellCmd(cmd_to_run,print_cmd,print_output)
		return

	def forward(self,add,model_in,data_in):
		"""Forward operator"""
		self.checkDomainRange(model_in,data_in)
		model = model_in
		data  = data_in

		#Apply operator to temporary file if necessary to add to given data vector
		sc = 0.
		if(add):
			data = data_in.clone()
			sc = 1.

		#If not out-of-core vectors convert them
		if(not isinstance(model,pyvec.vectorOC)):
			model=pyvec.vectorOC(model)
		if(not isinstance(data,pyvec.vectorOC)):
			data=pyvec.vectorOC(data)

		#Setting input/output and running the operator
		self.set_input_output_fwd(input=model.vecfile,output=data.vecfile)
		self.run_operator(self.cmd_fwd)

		#Converting temporary data vector back to correct vector type
		if(type(data) != type(data_in)):
			data_tmp = data_in.__new__(type(data_in))
			data_tmp.__init__(data)
			del data
			data = data_tmp
		#Adding data to input data vector
		data_in.scaleAdd(data,sc1=sc)
		return

	def adjoint(self,add,model_in,data_in):
		"""Adjoint operator"""
		self.checkDomainRange(model_in,data_in)

		model = model_in
		data  = data_in

		#Apply operator to temporary file if necessary to add to given data vector
		sc = 0.
		if(add):
			model = model_in.clone()
			sc = 1.

		#If not out-of-core vectors convert them
		if(not isinstance(model,pyvec.vectorOC)):
			model=pyvec.vectorOC(model)
		if(not isinstance(data,pyvec.vectorOC)):
			data=pyvec.vectorOC(data)

		#Setting input/output and running the operator
		self.set_input_output_adj(input=data.vecfile,output=model.vecfile)
		self.run_operator(self.cmd_adj)

		#Converting temporary data vector back to correct vector type
		if(type(model) != type(model_in)):
			model_tmp = model_in.__new__(type(model_in))
			model_tmp.__init__(model)
			del model
			model = model_tmp

		#Adding data to input data vector
		model_in.scaleAdd(model,sc1=sc)
		return
